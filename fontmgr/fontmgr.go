// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The font manager surface: family enumeration, family/style/character matching, style sets, and load-from-data, built
// over go-text/typesetting's fontscan index for the system manager (Default) and over caller-supplied font data for an
// in-memory one (NewFromData). A face is therefore either file-backed or data-backed, and each of its lazy reads — the
// display metadata, the rune-coverage probe, the typeface load — has a lane for both. Font metadata is
// host-independent: footprints (file location, normalized family, rune coverage, language coverage, aspect) group into
// families; per-face display names, style names, and font styles come from a lightweight table read
// (font.DescribeFaceFile/Data) on demand; style matching uses the CSS3 style-matching scoring in css3Score; character
// fallback probes the faces' cmap-derived rune coverage with BCP-47 hints ordered least-to-most significant, then
// verifies the chosen face really maps the character through its own cmap before answering — with another table-only
// read (font.FaceCoversRuneFile/Data), so a rejected candidate never becomes a fully parsed, permanently cached font.
//
// This is one host-independent implementation where a platform library would have three hosts (CoreText, fontconfig,
// DirectWrite), so its behavior is the documented font-manager contract plus the majority host behavior rather than any
// single host byte-for-byte. The system inventory is fontscan's on-disk scan, so it can include families a host has not
// activated and miss host-virtual families; hidden (dot-prefixed) families are excluded from enumeration but stay
// matchable by name and reachable through character fallback; enumeration is ordered by normalized family name; family
// lookup is case- and space-insensitive with no fontconfig-style alias substitution; selectStyleCSS3 is the one
// style-distance algorithm on every platform; and when nothing covers a character, matchFamilyStyleCharacter returns
// nil per the documented contract.

package fontmgr

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/fontscan"
	"github.com/go-text/typesetting/language"
	"github.com/richardwilkes/canvas/font"
)

// Manager is an immutable view of a font collection. All methods are safe for concurrent use. The zero value is an
// empty manager (no families); use Default for the system manager.
type Manager struct {
	byKey      map[string]*Family
	families   []*Family // sorted by normalized family key
	notHidden  []*Family // families minus the hidden (dot-prefixed) ones
	listed     []*Family // notHidden minus the families that can name no face — the enumeration list
	all        []*faceRec
	listedOnce sync.Once
}

// Family is one font family: the set of faces whose (normalized) family name matches.
type Family struct {
	key      string
	name     string
	faces    []*faceRec
	nameOnce sync.Once
}

// faceRec is one face known to the manager. Display metadata (family/style names, the OS/2-derived style) and the
// typeface itself load lazily and are cached for the life of the manager (cache aggressively). Exactly one of path and
// data holds the face's bytes: a scanned face re-reads its file for every lazy read, a face supplied to NewFromData
// reads the blob it came with.
type faceRec struct {
	tf       *font.Typeface
	key      string
	path     string // the font file, for a scanned face; "" for a data-backed one
	data     []byte // the font file's bytes, for a data-backed face; nil for a scanned one
	info     font.FaceInfo
	runes    fontscan.RuneSet
	langs    fontscan.LangSet
	index    int
	approx   font.Style
	infoOK   bool
	infoOnce sync.Once
	tfOnce   sync.Once

	// The bounded memo behind covers: the last coverMemoSize distinct runes probed, and their verdicts. coverLen is how
	// many entries are live (so rune 0 is a value like any other) and coverNext is the slot the next distinct rune
	// overwrites, round-robin.
	coverMu    sync.Mutex
	coverRunes [coverMemoSize]rune
	coverVals  [coverMemoSize]bool
	coverLen   int
	coverNext  int
}

// coverMemoSize is how many distinct runes one face remembers its cmap verdict for. Each probe is a file open and a
// cmap parse, and a fallback scan asks every claiming face about the same rune several times over — once per tier, and
// again per BCP-47 tag — so remembering only the rune in flight, which is what a single-entry memo does, still costs
// the whole inventory a fresh parse for every *distinct* character a run falls back on. Sixteen covers the interleaving
// a real run produces (a document mixes a handful of uncovered characters, not hundreds) at a fixed 96 bytes per face,
// allocated with the record and never grown — which is what rules out a map with an eviction policy here: the memo has
// to stay negligible beside the footprint data it sits in, and a system inventory holds thousands of these.
const coverMemoSize = 16

// faceInfo returns the lazily-loaded lightweight description (family/style names + style).
func (f *faceRec) faceInfo() (font.FaceInfo, bool) {
	f.infoOnce.Do(func() {
		var err error
		if f.path != "" {
			f.info, err = font.DescribeFaceFile(f.path, f.index)
		} else {
			f.info, err = font.DescribeFaceData(f.data, f.index)
		}
		f.infoOK = err == nil
	})
	return f.info, f.infoOK
}

// style returns the face's font style: the OS/2-derived style from the table read when available (always equal to the
// style of the typeface this face creates), else the footprint's approximate aspect.
func (f *faceRec) style() font.Style {
	if info, ok := f.faceInfo(); ok {
		return info.Style
	}
	return f.approx
}

func (f *faceRec) styleName() string {
	info, _ := f.faceInfo()
	return info.StyleName
}

// symbolPUAPage is the private-use page a symbol-encoded cmap (Microsoft platform 3, encoding 0) keys its characters
// under: such a font carries nothing below U+F000, and every resolver duplicates the U+F000–U+F0FF range at
// U+0000–U+00FF — Windows, HarfBuzz, and go-text's remaperSymbol, which is what Typeface.UnicharToGlyph and the cmap
// probe behind covers both go through.
const symbolPUAPage = 0xF000

// claims reports whether the face's recorded rune coverage can account for r. Both coverage sets a face can carry — a
// scan footprint's and the cmap-derived one NewFromData builds — are built by iterating the *raw* cmap subtable, while
// every resolution of r goes through go-text's remapping wrappers, whose added mappings the iteration deliberately does
// not include (see the "the Iter() and RuneRanges() method does not include the additional mapping" note above
// typesetting's remaperSymbol). So a set answering only for the code points it literally holds cannot see a symbol
// font's ASCII/Latin-1 coverage at all: Wingdings maps 'a' to a real glyph through the U+F061 entry its set does hold,
// and a filter asking only about 'a' vetoes the face before covers is ever asked. Asking about the U+F000-page
// character alongside r keeps the filter a superset of what the resolvers answer, which is all it has to be — a face
// claiming U+F0xx without a symbol cmap merely reaches covers, which rejects it.
func (f *faceRec) claims(r rune) bool {
	return f.runes.Contains(r) || (r >= 0 && r <= 0xFF && f.runes.Contains(symbolPUAPage+r))
}

// covers reports whether the face's own cmap maps r to a real glyph, reading only the cmap (and OS/2, which selects its
// encoding) instead of loading the typeface: the character-fallback scans reject most of their candidates, and a
// typeface, once loaded, is cached — with the whole font file inside it — for the life of the manager, so verifying
// through typeface() made one MatchFamilyStyleCharacter call retain a large fraction of the system inventory. Only the
// answering candidate is loaded now.
//
// The answer is memoized for the last coverMemoSize distinct runes, which is what the scans re-ask: a face reached
// through the default-family tier is re-probed by the all-visible-families tier, a face in a BCP-47 restricted set is
// re-probed by the unrestricted scan, and a run of text falls back over the same handful of characters again and again,
// interleaved. Nothing but those runes and their verdicts is retained (see coverMemoSize).
func (f *faceRec) covers(r rune) bool {
	f.coverMu.Lock()
	defer f.coverMu.Unlock()
	for i := range f.coverLen {
		if f.coverRunes[i] == r {
			return f.coverVals[i]
		}
	}
	var val bool
	if f.path != "" {
		val = font.FaceCoversRuneFile(f.path, f.index, r)
	} else {
		val = font.FaceCoversRuneData(f.data, f.index, r)
	}
	f.coverRunes[f.coverNext] = r
	f.coverVals[f.coverNext] = val
	f.coverNext = (f.coverNext + 1) % coverMemoSize
	if f.coverLen < coverMemoSize {
		f.coverLen++
	}
	return val
}

// typeface loads (and caches) the full typeface for this face; nil if the file no longer parses.
func (f *faceRec) typeface() *font.Typeface {
	f.tfOnce.Do(func() {
		var err error
		if f.path != "" {
			f.tf, err = font.NewTypefaceFromFile(f.path, f.index)
		} else {
			f.tf, err = font.NewTypefaceFromData(f.data, f.index)
		}
		if err != nil {
			f.tf = nil
		}
	})
	return f.tf
}

// Name returns the family's display name: the first face's described family (which normalizes back to the grouping key
// by construction), falling back to the normalized key if no face parses. The fallback is a last resort for a family
// reached by name — enumeration drops such a family instead of showing the key (see Manager.enumerated).
func (f *Family) Name() string {
	if name := f.describedName(); name != "" {
		return name
	}
	return f.key
}

// describedName returns the display family name of the family's first describable face, "" when no face of the family
// can be described at all. Computed once and cached: a font picker asks for every family's name, and the answer is the
// same table read the enumeration filter needs.
func (f *Family) describedName() string {
	f.nameOnce.Do(func() {
		for _, face := range f.faces {
			if info, ok := face.faceInfo(); ok && info.Family != "" {
				f.name = info.Family
				return
			}
		}
	})
	return f.name
}

// scanLogLines is how many of fontscan's log lines a scanLogger keeps. fontscan logs one line per unreadable font file
// as well as its summary lines, so the whole stream is unbounded on a broken machine while the last few lines are what
// name the failure.
const scanLogLines = 8

// scanLogger captures fontscan's informational logging rather than discarding it. fontscan reports its diagnostics
// through a logger instead of through its error, and a library has no business writing to a process-wide log stream, so
// the lines are held here and folded into the error the scan surfaces — the one place they are of use. Safe for
// concurrent use because fontscan scans font directories in parallel.
type scanLogger struct {
	lines []string
	mu    sync.Mutex
}

func (l *scanLogger) Printf(format string, args ...any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lines = append(l.lines, strings.TrimSpace(fmt.Sprintf(format, args...)))
	if len(l.lines) > scanLogLines {
		l.lines = l.lines[len(l.lines)-scanLogLines:]
	}
}

// tail returns the retained lines, oldest first, as one semicolon-separated string; "" when nothing was logged.
func (l *scanLogger) tail() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return strings.Join(l.lines, "; ")
}

// memoizedScan holds the process's one system scan and its result. The scan function and the cache-directory resolver
// are fields rather than direct calls so that the memoization contract itself — one attempt, one answer, both entry
// points agreeing — is testable without touching the machine's fonts or its cache directory, exactly as scanSystemFonts
// takes them as parameters for the same reason.
type memoizedScan struct {
	scan     func(fontscan.Logger, string) ([]fontscan.Footprint, error)
	cacheDir func() string
	mgr      *Manager
	err      error
	once     sync.Once
}

// resolve performs the one scan, on the first call; mgr and err hold its result for every caller after that.
func (s *memoizedScan) resolve() {
	s.once.Do(func() { s.mgr, s.err = scanSystemFonts(s.scan, s.cacheDir()) })
}

// systemDefault is the process's system scan, behind Default and DefaultWithError.
var systemDefault = &memoizedScan{scan: fontscan.SystemFonts, cacheDir: systemFontCacheDir}

// Default returns the process-wide system font manager. The first call scans the system fonts through fontscan (an
// on-disk index cache makes later runs cheap) and the manager it builds — empty, never nil, when the scan fails — is
// memoized for the life of the process. Use DefaultWithError when a failed scan has to be reported: that is the only
// place the reason is visible.
func Default() *Manager {
	systemDefault.resolve()
	return systemDefault.mgr
}

// DefaultWithError returns what Default returns, plus the error from the one system scan the process performs (nil when
// it succeeded; a scan that finds no usable face is an error even where fontscan reported none). Both are memoized, so
// the reason a process has no system fonts stays available to every caller rather than only to whoever called first. A
// non-nil error always comes with an empty, non-nil manager.
//
// The scan is never retried. fontscan guards its system index with a package-level sync.Once, so the first attempt in
// the process is the only one that can run: a second fontscan.SystemFonts call after a failed first returns an empty
// index and a *nil* error, which would replace a diagnosable failure with a silent one. Making the single attempt
// succeed is therefore what systemFontCacheDir is for.
func DefaultWithError() (*Manager, error) {
	systemDefault.resolve()
	return systemDefault.mgr, systemDefault.err
}

// scanSystemFonts runs one system scan and groups its footprints into a manager. The scan function and the cache
// directory are parameters so the failure and empty-result paths are testable without touching the machine's fonts or
// its cache directory; Default passes fontscan.SystemFonts and systemFontCacheDir().
func scanSystemFonts(scan func(fontscan.Logger, string) ([]fontscan.Footprint, error), cacheDir string) (*Manager, error) {
	logger := &scanLogger{}
	fps, err := scan(logger, cacheDir)
	if err != nil {
		return newManager(nil), scanError(fmt.Errorf("fontmgr: system font scan failed: %w", err), logger)
	}
	recs := make([]*faceRec, 0, len(fps))
	for i := range fps {
		fp := &fps[i]
		if fp.Family == "" {
			continue
		}
		recs = append(recs, &faceRec{
			key:    fp.Family,
			path:   fp.Location.File,
			index:  int(fp.Location.Index),
			runes:  fp.Runes,
			langs:  fp.Langs,
			approx: styleFromAspect(fp.Aspect),
		})
	}
	if len(recs) == 0 {
		// A successful scan carries at least one valid face (fontscan asserts that before caching its index), so an
		// empty result means either that every footprint lacked a family name or that another caller in this process
		// already consumed fontscan's one-shot initialization with a failure — the case that hands back an empty index
		// and no error at all.
		return newManager(nil), scanError(errors.New("fontmgr: system font scan found no usable faces"), logger)
	}
	return newManager(recs), nil
}

// scanError appends whatever fontscan logged to err, since the failure's actual cause (an unreadable directory, a font
// that would not parse) is usually only in the log.
func scanError(err error, logger *scanLogger) error {
	if tail := logger.tail(); tail != "" {
		return fmt.Errorf("%w [fontscan: %s]", err, tail)
	}
	return err
}

// systemFontCacheDir picks the directory fontscan serializes its font index cache into. Left to choose for itself
// fontscan uses os.UserCacheDir, and it discards a *fully successful* scan when it cannot write the index there
// (refreshSystemFontsIndex turns serializeToFile's error into the SystemFonts error), so a process with HOME unset or
// read-only — a systemd unit, a container, a sandboxed app — ends up with no system fonts at all rather than with
// uncached ones. Resolving the directory here lets such a scan fall back to the temporary directory, which is writable
// in practically every environment the user cache dir is not. The user cache dir stays the first choice and is passed
// through unchanged, so the ordinary case reads and writes exactly the file fontscan would have picked on its own.
//
// The fallback is a per-user subdirectory of the temporary directory: a shared /tmp is sticky, so one user's index
// cache must not be a file another user's process then fails to rewrite. "" is returned when neither candidate is
// writable, which leaves fontscan its own choice and its own error.
func systemFontCacheDir() string { return chooseFontCacheDir(os.UserCacheDir, os.TempDir) }

func chooseFontCacheDir(userCacheDir func() (string, error), tempDir func() string) string {
	if dir, err := userCacheDir(); err == nil && cacheDirWritable(dir) {
		return dir
	}
	name := "canvas-fontscan"
	if uid := os.Getuid(); uid >= 0 { // -1 on Windows, whose temporary directory is per-user already
		name += "-" + strconv.Itoa(uid)
	}
	if dir := filepath.Join(tempDir(), name); cacheDirWritable(dir) {
		return dir
	}
	return ""
}

// cacheDirWritable reports whether an index cache can be written into dir, creating it as fontscan's own
// serializeToFile would. There is no portable permission query, so the check is a probe file: creating and removing one
// is what the write itself will do.
func cacheDirWritable(dir string) bool {
	if dir == "" {
		return false
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return false
	}
	f, err := os.CreateTemp(dir, "canvas-fontcache-probe-*")
	if err != nil {
		return false
	}
	name := f.Name()
	closeErr := f.Close()
	return os.Remove(name) == nil && closeErr == nil
}

// NewFromData builds a manager over in-memory font data instead of a system scan: each blob is one font file (anything
// the sfnt reader accepts — TTF, OTF, TTC, WOFF), every face of every blob becomes a matchable face, and the faces
// group into families by the same normalized family name the scan groups by. Nothing touches the filesystem and no
// system font is involved, so an application's own fonts (a go:embed corpus, a downloaded file) get the same family
// enumeration, style matching, and character fallback the system manager offers. A blob that does not parse, and any
// face carrying no family name, is skipped; the result is never nil, and is empty when nothing usable was supplied.
//
// Two things differ from a scanned face. Rune coverage is read from each face's own cmap here, so it holds exactly the
// code points that cmap maps to a real glyph, where a scan footprint also counts the entries that map to .notdef. That
// removes the over-claim, not the under-claim: both sets are built by iterating the raw subtable, so neither can see
// the mappings go-text's remapping wrappers add on top of it, and a symbol-encoded face's set therefore holds
// U+F020–U+F0FF and nothing below it while the face really does resolve U+0020–U+00FF through that page. The candidate
// filter widens for exactly that (see faceRec.claims); the set is a filter in front of the cmap probe, never the last
// word on what a face covers. Language coverage is left empty — fontscan derives its language sets while indexing, from
// data this package cannot reach — so a BCP-47 hint never restricts a candidate set to these faces, though the
// unrestricted scan still reaches all of them.
//
// The blobs are retained for the life of the manager (each face parses lazily out of its own blob, as a scanned face
// re-reads its file) and must not be modified afterwards.
func NewFromData(data ...[]byte) *Manager {
	var recs []*faceRec
	for _, blob := range data {
		lds, err := opentype.NewLoaders(bytes.NewReader(blob))
		if err != nil {
			continue
		}
		for index := range lds {
			info, err2 := font.DescribeFaceData(blob, index)
			if err2 != nil || info.Family == "" {
				continue
			}
			var runes fontscan.RuneSet
			for _, r := range font.FaceRunesData(blob, index) {
				runes.Add(r)
			}
			recs = append(recs, &faceRec{
				key:    tsfont.NormalizeFamily(info.Family),
				data:   blob,
				index:  index,
				runes:  runes,
				approx: info.Style,
			})
		}
	}
	return newManager(recs)
}

// newManager groups faces into families (sorted by normalized key; faces keep their scan order within a family).
func newManager(recs []*faceRec) *Manager {
	m := &Manager{byKey: make(map[string]*Family)}
	for _, rec := range recs {
		fam := m.byKey[rec.key]
		if fam == nil {
			fam = &Family{key: rec.key}
			m.byKey[rec.key] = fam
			m.families = append(m.families, fam)
		}
		fam.faces = append(fam.faces, rec)
	}
	sort.Slice(m.families, func(i, j int) bool { return m.families[i].key < m.families[j].key })
	for _, fam := range m.families {
		m.all = append(m.all, fam.faces...)
		// Hidden families (macOS's dot-prefixed UI fonts) stay matchable by name and reachable through character
		// fallback, but are excluded from enumeration, matching CTFontManagerCopyAvailableFontFamilyNames.
		if !strings.HasPrefix(fam.key, ".") {
			m.notHidden = append(m.notHidden, fam)
		}
	}
	return m
}

// enumerated returns the enumeration list: the visible families that can actually name themselves. A family none of
// whose faces this port can even describe cannot produce a typeface either, so listing it offers a font picker an entry
// that selects nothing and — since Name() has only the normalized grouping key left to fall back on — labels it with a
// lowercased, space-stripped key rather than a display name (stock macOS ships one, NISC18030.ttf, which carries no
// head table). Both halves of that are the port's own bookkeeping leaking out, so such a family is dropped from the
// list while staying matchable by name and reachable through character fallback, exactly as a hidden family is.
//
// The filter is computed on the first enumeration and cached. It costs one lightweight table read per family, which is
// the same read the caller's own FamilyName call makes and which faceRec memoizes for the life of the manager.
func (m *Manager) enumerated() []*Family {
	m.listedOnce.Do(func() {
		for _, fam := range m.notHidden {
			if fam.describedName() != "" {
				m.listed = append(m.listed, fam)
			}
		}
	})
	return m.listed
}

// CountFamilies returns the number of enumerable families.
func (m *Manager) CountFamilies() int { return len(m.enumerated()) }

// FamilyName returns the display name of the index-th family, "" out of range.
func (m *Manager) FamilyName(index int) string {
	listed := m.enumerated()
	if index < 0 || index >= len(listed) {
		return ""
	}
	return listed[index].Name()
}

// MatchFamily returns the style set for the named family (normalized, so the lookup is case- and space-insensitive like
// the platform hosts'). An empty name requests the platform default family, resolved exactly as MatchFamilyStyle
// resolves it (the documented contract is the same for both: a null family name asks for the default system family), so
// MatchFamily("").MatchStyle(s) and MatchFamilyStyle("", s) always agree. Unknown names — and an empty name in a
// manager with no families at all — yield an empty set, never nil.
func (m *Manager) MatchFamily(familyName string) *StyleSet {
	fam := m.lookupOrDefault(familyName)
	if fam == nil {
		return &StyleSet{}
	}
	return &StyleSet{faces: fam.faces}
}

// MatchFamilyStyle returns the best style match within the named family, nil when the family is unknown (both the
// CoreText and fontconfig hosts return null after their family-name check). An empty family name requests the platform
// default family, as the hosts do for a null familyName.
func (m *Manager) MatchFamilyStyle(familyName string, style font.Style) *font.Typeface {
	fam := m.lookupOrDefault(familyName)
	if fam == nil {
		return nil
	}
	return (&StyleSet{faces: fam.faces}).MatchStyle(style)
}

func (m *Manager) lookupOrDefault(familyName string) *Family {
	if familyName != "" {
		return m.byKey[tsfont.NormalizeFamily(familyName)]
	}
	for _, name := range defaultFamilies() {
		if fam := m.byKey[tsfont.NormalizeFamily(name)]; fam != nil {
			return fam
		}
	}
	// The last resorts are the first enumerable family and then, for a manager whose every family is hidden or
	// undescribable, the first family of all: a default worth handing back is one a caller could also have picked out
	// of the enumeration, but "something" still beats nothing.
	if listed := m.enumerated(); len(listed) != 0 {
		return listed[0]
	}
	if len(m.families) != 0 {
		return m.families[0]
	}
	return nil
}

// defaultFamilies is the platform default-family search order used when no family name is given, approximating each
// host's default font (CoreText: Helvetica; DirectWrite: Segoe UI; fontconfig: its sans-serif alias).
func defaultFamilies() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"Helvetica", "Helvetica Neue", "Arial"}
	case "windows":
		return []string{"Segoe UI", "Tahoma", "Arial"}
	default:
		return []string{"DejaVu Sans", "Liberation Sans", "Noto Sans", "FreeSans", "Arial"}
	}
}

// MatchFamilyStyleCharacter finds a typeface covering character, preferring the named family, then fonts covering the
// most significant matching BCP-47 language (bcp47[0] is the least significant, bcp47[len-1] the most), then the
// closest style. Returns nil for invalid code points (surrogates, out of range — the CoreText host's CFString
// rejection) and when no known font covers the character (the fontconfig host's FontContainsCharacter check; the
// CoreText host instead returns the family font). "Covers" means the face's cmap maps the character to a real glyph:
// candidates whose footprint claims the rune but whose cmap yields glyph 0 are skipped (matchCovering), so the returned
// face always maps character through its own cmap.
func (m *Manager) MatchFamilyStyleCharacter(familyName string, style font.Style, bcp47 []string, character int32) *font.Typeface {
	if character < 0 || character > 0x10FFFF || (character >= 0xD800 && character <= 0xDFFF) {
		return nil
	}
	r := character
	// The named family wins when any of its faces covers the character (the CoreText host resolves the styled family
	// font first and CTFontCreateForString returns it unchanged when it covers the string).
	if familyName != "" {
		if fam := m.byKey[tsfont.NormalizeFamily(familyName)]; fam != nil {
			if tf := matchCovering(fam.faces, style, r, false); tf != nil {
				return tf
			}
		}
	}
	// BCP-47 pass: most significant tag last. Each tag (scanning back to front) restricts the candidate set to the fonts
	// claiming both the rune and the language; the first tag whose restricted set yields a verified covering face wins.
	// A tag with candidates that all fail verification (a lying footprint) or fail to load falls through to the next
	// less-significant tag and finally to the unrestricted scan, as the named-family pass above does — the contract
	// promises nil only when no known font covers the character.
	for i := len(bcp47) - 1; i >= 0; i-- {
		id, ok := language.NewLangID(language.NewLanguage(bcp47[i]))
		if !ok {
			continue
		}
		var candidates []*faceRec
		for _, f := range m.all {
			if f.claims(r) && f.langs.Contains(id) {
				candidates = append(candidates, f)
			}
		}
		if len(candidates) != 0 {
			if tf := m.matchCoveringTiered(candidates, style, r); tf != nil {
				return tf
			}
		}
	}
	return m.matchCoveringTiered(m.all, style, r)
}

// matchCoveringTiered is the cross-family fallback selection: default-family faces are preferred over other visible
// families, which are preferred over hidden (dot-prefixed) families — approximating the platform cascade lists, which
// never surface the hidden UI fonts fontscan's raw index includes — with the CSS3 style score ordering within each
// tier. defaultFamilies is a search order rather than a pool, so each present default family is its own tier, in that
// order: an approximate-style face of the first default family outranks an exact-style face of the second, exactly as
// lookupOrDefault's first-family-found-wins does for family lookup.
func (m *Manager) matchCoveringTiered(faces []*faceRec, style font.Style, r rune) *font.Typeface {
	var visible, hidden []*faceRec
	for _, f := range faces {
		if !f.claims(r) {
			continue
		}
		if strings.HasPrefix(f.key, ".") {
			hidden = append(hidden, f)
		} else {
			visible = append(visible, f)
		}
	}
	tiers := append(m.defaultFamilyTiers(visible), visible, hidden)
	for _, candidates := range tiers {
		if tf := matchCovering(candidates, style, r, true); tf != nil {
			return tf
		}
	}
	return nil
}

// defaultFamilyTiers returns one tier per present default family, in defaultFamilies order, holding only that family's
// faces that are part of the candidate set: a default family may have been excluded by a BCP-47 restriction or may not
// claim r at all (matchCovering re-checks the claim, and the face's own cmap has the final word).
func (m *Manager) defaultFamilyTiers(candidates []*faceRec) [][]*faceRec {
	var families []*Family
	for _, name := range defaultFamilies() {
		if fam := m.byKey[tsfont.NormalizeFamily(name)]; fam != nil {
			families = append(families, fam)
		}
	}
	if len(families) == 0 {
		return nil
	}
	allowed := make(map[*faceRec]bool, len(candidates))
	for _, f := range candidates {
		allowed[f] = true
	}
	tiers := make([][]*faceRec, 0, len(families)+2) // the caller appends the visible and hidden tiers
	for _, fam := range families {
		var kept []*faceRec
		for _, f := range fam.faces {
			if allowed[f] {
				kept = append(kept, f)
			}
		}
		if len(kept) != 0 {
			tiers = append(tiers, kept)
		}
	}
	return tiers
}

// matchCovering returns the best css3-scored face among those covering r, nil when none cover (or none load —
// lower-scored covering faces are tried in rank order, as in StyleSet.MatchStyle). approx selects the footprint-derived
// style (no I/O — used for the cross-family fallback scans) over the exact table-read style (used within a single
// family).
//
// A recorded rune set is approximate in both directions, so it decides nothing on its own.
//
// It overcounts: go-text's scanner counts cmap entries that map to glyph 0 (fontforge-built fonts commonly carry
// U+0000 and U+FFFF segments mapping to .notdef — ubuntu's DejaVuSans-ExtraLight does), and no host treats a .notdef
// mapping as coverage: FreeType's charcode iteration skips glyph-0 entries, so fontconfig charsets never contain them,
// and the CoreText/DirectWrite cmap lookups yield the missing glyph. So for a candidate the set does put forward, the
// face's own cmap is the final word — it only answers when it really maps r. That verdict comes from faceRec's
// cmap-only probe (covers), so the walk loads a typeface only for the candidate that is about to be returned; a
// character that many footprints claim and few really map used to load, and permanently cache, one full font per
// rejection.
//
// It also undercounts, for the remapped cmaps a set built by raw iteration cannot describe (a symbol-encoded face maps
// 'a' through its U+F061 entry, and holds nothing at 'a' at all). A set that only overcounted would make membership a
// sound filter; one that undercounts too would let it veto faces the resolvers would rescue, before covers is ever
// asked. So the filter is faceRec.claims rather than membership itself: claims widens the question by the remap, and
// the set may then only put a face in front of covers, never keep it out.
func matchCovering(faces []*faceRec, pattern font.Style, r rune, approx bool) *font.Typeface {
	var covering []*faceRec
	for _, f := range faces {
		if f.claims(r) {
			covering = append(covering, f)
		}
	}
	styleAt := func(i int) font.Style {
		if approx {
			return covering[i].approx
		}
		return covering[i].style()
	}
	var match *font.Typeface
	selectStyleCSS3(pattern, len(covering), styleAt, func(idx int) bool {
		f := covering[idx]
		if !f.covers(r) {
			return false
		}
		// The probe already answered the cmap question; re-asking the loaded typeface costs one lookup and keeps the
		// contract ("the returned face always maps character through its own cmap") true of the object handed back, not
		// just of the file it came from.
		tf := f.typeface()
		if tf == nil || tf.UnicharToGlyph(r) == 0 {
			return false
		}
		match = tf
		return true
	})
	return match
}

// MakeFromData loads a typeface from font data; nil when the data is not a usable font or the collection index is out
// of range.
func (m *Manager) MakeFromData(data []byte, index int) *font.Typeface {
	tf, err := font.NewTypefaceFromData(data, index)
	if err != nil {
		return nil
	}
	return tf
}
