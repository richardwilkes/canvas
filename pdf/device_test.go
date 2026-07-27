// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"bytes"
	"compress/zlib"
	"io"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stream"
)

// renderPDF draws through a device-backed page and returns the serialized PDF bytes.
func renderPDF(t *testing.T, w, h float32, draw func(c *canvas.Canvas)) []byte {
	t.Helper()
	var buf stream.MemoryWStream
	doc := NewDocument(&buf, DefaultMetadata())
	c := doc.BeginPageCanvas(w, h)
	draw(c)
	doc.EndPage()
	doc.Close()
	return append([]byte(nil), buf.Bytes()...)
}

// allStreamContents returns every stream object's payload, inflating FlateDecode streams.
func allStreamContents(data []byte) []string {
	var out []string
	rest := data
	for {
		si := bytes.Index(rest, []byte(" stream\n"))
		if si < 0 {
			break
		}
		ei := bytes.Index(rest[si:], []byte("\nendstream"))
		if ei < 0 {
			break
		}
		payload := rest[si+len(" stream\n") : si+ei]
		dictStart := bytes.LastIndex(rest[:si], []byte("obj\n"))
		if dictStart >= 0 && bytes.Contains(rest[dictStart:si], []byte("/FlateDecode")) {
			if r, err := zlib.NewReader(bytes.NewReader(payload)); err == nil {
				if dec, readErr := io.ReadAll(r); readErr == nil {
					payload = dec
				}
			}
		}
		out = append(out, string(payload))
		rest = rest[si+ei+len("\nendstream"):]
	}
	return out
}

// pageContent returns the (first) content stream — the one prefixed by the initial "cm" transform.
func pageContent(t *testing.T, data []byte) string {
	t.Helper()
	for _, s := range allStreamContents(data) {
		if strings.Contains(s, " cm\n") {
			return s
		}
	}
	// A page with an identity initial transform (rasterDPI produces one only for a 0-height page) still has a content
	// stream; fall back to the last stream.
	all := allStreamContents(data)
	if len(all) > 0 {
		return all[len(all)-1]
	}
	t.Fatal("no content stream found")
	return ""
}

func mustContain(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Errorf("content missing %q\n--- content ---\n%s", needle, haystack)
	}
}

// TestPopulateGraphicStateEntryShaderRouting pins which shaders populateGraphicStateEntry folds into entry.color and
// which become a /Pattern resource in entry.shaderIndex. Only a ColorShader (and no shader at all) collapses to a
// color; gradient, image, and generic-fallback shaders all route through makeShader and install a pattern.
func TestPopulateGraphicStateEntryShaderRouting(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 0x40, 0x80, 0xC0, 0xFF)
	stops := []colorcore.Color{colorcore.ARGB(255, 255, 0, 0), colorcore.ARGB(255, 0, 0, 255)}
	cases := []struct {
		shader      shaders.Shader
		name        string
		wantPattern bool
	}{
		{name: "none", shader: nil, wantPattern: false},
		{name: "color", shader: shaders.NewColor(colorcore.ARGB(255, 10, 20, 30)), wantPattern: false},
		{
			name:        "gradient",
			shader:      shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 32}, stops, nil, shaders.TileClamp, nil),
			wantPattern: true,
		},
		{
			name:        "image",
			shader:      shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, shaders.SamplingOptions{}, nil),
			wantPattern: true,
		},
		{
			name: "fallback",
			shader: shaders.NewBlend(raster.BlendPlus, shaders.NewColor(colorcore.ARGB(255, 200, 40, 40)),
				shaders.NewColor(colorcore.ARGB(255, 40, 40, 200))),
			wantPattern: true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var buf stream.MemoryWStream
			doc := NewDocument(&buf, DefaultMetadata())
			d := newDevice(geom.ISize{Width: 32, Height: 32}, doc, geom.IdentityMatrix())
			paint := canvas.NewPaint()
			paint.Color = colorcore.ARGB(255, 200, 30, 40)
			paint.Shader = c.shader
			matrix := geom.IdentityMatrix()
			entry := defaultGSEntry()
			d.populateGraphicStateEntry(&matrix, nil, paint, 1, &entry)
			if got := entry.shaderIndex >= 0; got != c.wantPattern {
				t.Fatalf("shaderIndex = %d (pattern=%v), want pattern=%v", entry.shaderIndex, got, c.wantPattern)
			}
			if !c.wantPattern {
				return
			}
			// The installed pattern is registered as a device shader resource, so the page's /Pattern dict names it.
			if _, ok := d.shaderResources[IndirectReference{value: int32(entry.shaderIndex)}]; !ok {
				t.Errorf("pattern object %d is not in the device's shader resources %v",
					entry.shaderIndex, d.shaderResources)
			}
		})
	}
}

// TestNonGradientShaderPaintsWithItsPattern proves an image shader's pattern (not the paint color) is what the page
// content paints with: the paint color would have shown up as an "rg" fill.
func TestNonGradientShaderPaintsWithItsPattern(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 0x40, 0x80, 0xC0, 0xFF)
	paint := canvas.NewPaint()
	paint.Color = colorcore.ARGB(255, 200, 30, 40) // .7843 .1176 .1569
	paint.Shader = shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, shaders.SamplingOptions{}, nil)
	data := renderPDFUncompressed(t, 40, 40, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 40, 40), paint)
	})
	validatePDF(t, data)

	// The page content stream is the one carrying the page's top-left → bottom-left flip; the pattern cell's stream
	// also holds a "cm", so it cannot be picked by that alone.
	var content string
	for _, s := range allStreamContents(data) {
		if strings.Contains(s, "1 0 0 -1 0 40 cm\n") {
			content = s
			break
		}
	}
	if content == "" {
		t.Fatalf("no page content stream\nstreams=%v", allStreamContents(data))
	}
	mustContain(t, content, "/Pattern cs")
	mustContain(t, content, " scn\n")
	if strings.Contains(content, ".7843 .1176 .1569 rg") {
		t.Errorf("the image shader's fill fell back to the paint color\n--- content ---\n%s", content)
	}
}

func TestDeviceFilledRect(t *testing.T) {
	red := canvas.NewPaint()
	red.Color = colorcore.ARGB(255, 200, 30, 40)
	data := renderPDF(t, 200, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(10, 20, 110, 70), red)
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	// Top-left → bottom-left flip for a 100-unit-tall page.
	mustContain(t, content, "1 0 0 -1 0 100 cm\n")
	// A closed CW rect collapses to a single "re" (x, y, width, height) + fill.
	mustContain(t, content, "10 20 100 50 re\n")
	mustContain(t, content, "f\n")
	// Color: 200/255, 30/255, 40/255 with 4 significant digits, as both stroke and fill color.
	mustContain(t, content, ".7843 .1176 .1569 RG .7843 .1176 .1569 rg\n")
}

func TestDeviceStrokedPath(t *testing.T) {
	blue := canvas.NewPaint()
	blue.Color = colorcore.ARGB(255, 10, 20, 220)
	blue.Style = canvas.StyleStroke
	blue.StrokeWidth = 3
	data := renderPDF(t, 200, 200, func(c *canvas.Canvas) {
		p := (&path.Path{}).MoveTo(20, 20).LineTo(80, 90).LineTo(150, 30)
		c.DrawPath(p, blue)
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	mustContain(t, content, "20 20 m\n")
	mustContain(t, content, "80 90 l\n")
	mustContain(t, content, "150 30 l\n")
	mustContain(t, content, "S\n")
}

func TestDeviceClipRectEmitsRectClip(t *testing.T) {
	paint := canvas.NewPaint()
	data := renderPDF(t, 200, 200, func(c *canvas.Canvas) {
		c.Save()
		c.ClipRect(geom.RectLTRB(10, 10, 90, 90), raster.ClipIntersect, false)
		c.DrawRect(geom.RectLTRB(0, 0, 200, 200), paint)
		c.Restore()
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	// A rect clip is emitted as a rectangle + "W* n", inside a q/Q pair. The device-bounds outset is clamped away by
	// the intersection with the clip rect, so the emitted rect is the clip itself.
	mustContain(t, content, "q\n")
	mustContain(t, content, "10 10 80 80 re\n")
	mustContain(t, content, "W* n\n")
	mustContain(t, content, "Q\n")
}

func TestDeviceClipPathEmitsPathClip(t *testing.T) {
	paint := canvas.NewPaint()
	data := renderPDF(t, 200, 200, func(c *canvas.Canvas) {
		c.Save()
		clip := (&path.Path{}).MoveTo(10, 10).LineTo(100, 20).LineTo(50, 120)
		clip.Close()
		c.ClipPath(clip, raster.ClipIntersect, false)
		c.DrawRect(geom.RectLTRB(0, 0, 200, 200), paint)
		c.Restore()
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	// A non-rect clip emits a path terminated by the nonzero clip operator "W n".
	mustContain(t, content, "W n\n")
}

func TestDeviceGraphicStateCanonicalization(t *testing.T) {
	// Two opaque fills with different colors but the same alpha/blend must share one /ExtGState object.
	a := canvas.NewPaint()
	a.Color = colorcore.ARGB(255, 255, 0, 0)
	b := canvas.NewPaint()
	b.Color = colorcore.ARGB(255, 0, 255, 0)
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 10, 10), a)
		c.DrawRect(geom.RectLTRB(20, 20, 30, 30), b)
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	// Both draws reference the same graphic-state resource name (emitted once, since the state is shared).
	if n := strings.Count(content, " gs\n"); n != 1 {
		t.Errorf("expected the shared /ExtGState to be applied once, got %d gs operators\n%s", n, content)
	}
	// The ExtGState object carries an opaque fill alpha and the Normal blend mode.
	foundGS := false
	for _, s := range dictObjects(data) {
		if strings.Contains(s, "/ca 1") && strings.Contains(s, "/BM /Normal") {
			foundGS = true
		}
	}
	if !foundGS {
		t.Error("no fill /ExtGState with /ca 1 and /BM /Normal")
	}
}

func TestDeviceStrokeGraphicState(t *testing.T) {
	p := canvas.NewPaint()
	p.Style = canvas.StyleStroke
	p.StrokeWidth = 4
	p.MiterLimit = 3
	p.Cap = canvas.CapRound
	p.Join = canvas.JoinBevel
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawLine(10, 10, 90, 90, p)
	})
	validatePDF(t, data)
	found := false
	for _, s := range dictObjects(data) {
		if strings.Contains(s, "/LW 4") && strings.Contains(s, "/ML 3") &&
			strings.Contains(s, "/LC 1") && strings.Contains(s, "/LJ 2") &&
			strings.Contains(s, "/SA true") {
			found = true
		}
	}
	if !found {
		t.Error("no stroke /ExtGState with the expected LW/ML/LC/LJ/SA entries")
	}
}

func TestDeviceAlphaAndBlend(t *testing.T) {
	p := canvas.NewPaint()
	p.Color = colorcore.ARGB(128, 255, 0, 0)
	p.BlendMode = raster.BlendMultiply
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 50, 50), p)
	})
	validatePDF(t, data)
	found := false
	for _, s := range dictObjects(data) {
		if strings.Contains(s, "/BM /Multiply") && strings.Contains(s, "/ca .502") {
			found = true
		}
	}
	if !found {
		t.Errorf("no /ExtGState with /BM /Multiply and the ~0.5 alpha; objects=%v", dictObjects(data))
	}
}

func TestDevicePointsModes(t *testing.T) {
	p := canvas.NewPaint()
	p.Style = canvas.StyleStroke
	p.StrokeWidth = 2
	pts := []geom.Point{{X: 10, Y: 10}, {X: 40, Y: 20}, {X: 70, Y: 5}, {X: 90, Y: 60}}

	poly := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawPoints(canvas.PointModePolygon, pts, p)
	})
	validatePDF(t, poly)
	pc := pageContent(t, poly)
	mustContain(t, pc, "10 10 m\n")
	mustContain(t, pc, "40 20 l\n")
	mustContain(t, pc, "90 60 l\n")
	mustContain(t, pc, "S\n")

	lines := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawPoints(canvas.PointModeLines, pts, p)
	})
	validatePDF(t, lines)
	lc := pageContent(t, lines)
	// Two independent segments, each moveto+lineto+stroke.
	if n := strings.Count(lc, " m\n"); n != 2 {
		t.Errorf("kLines expected 2 movetos, got %d\n%s", n, lc)
	}
}

func TestDeviceEmptyPageHasNoContent(t *testing.T) {
	// A page with no draws still produces a valid one-page PDF.
	data := renderPDF(t, 50, 50, func(_ *canvas.Canvas) {})
	validatePDF(t, data)
}

func TestDeviceBalancedSaveRestore(t *testing.T) {
	p := canvas.NewPaint()
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.Save()
		c.ClipRect(geom.RectLTRB(10, 10, 90, 90), raster.ClipIntersect, false)
		c.DrawRect(geom.RectLTRB(0, 0, 100, 100), p)
		c.Save()
		c.ClipRect(geom.RectLTRB(20, 20, 80, 80), raster.ClipIntersect, false)
		c.DrawOval(geom.RectLTRB(25, 25, 75, 75), p)
		c.Restore()
		c.Restore()
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	if q, qq := strings.Count(content, "q\n"), strings.Count(content, "Q\n"); q != qq {
		t.Errorf("unbalanced q/Q: %d q vs %d Q\n%s", q, qq, content)
	}
}

func TestDeviceSaveLayerCompositesContent(t *testing.T) {
	// Before the compositing slice, the PDF drawDevice was a no-op and saveLayer content was silently dropped; it must
	// now survive inside a form XObject the page invokes.
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.SaveLayer(nil, nil)
		red := canvas.NewPaint()
		red.Color = colorcore.ARGB(255, 255, 0, 0)
		c.DrawRect(geom.RectLTRB(10, 10, 50, 50), red)
		c.Restore()
	})
	validatePDF(t, data)
	if !bytes.Contains(data, []byte("/Subtype /Form")) {
		t.Fatal("saveLayer content was dropped (no form XObject)")
	}
	joined := strings.Join(allStreamContents(data), "\n")
	mustContain(t, joined, "10 10 40 40 re\n") // the layer's red rect, inside the form XObject
	mustContain(t, joined, " Do\n")            // the page invokes the layer XObject
}

func TestDeviceSaveLayerAlphaTransparencyGroup(t *testing.T) {
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.SaveLayerAlpha(nil, 128)
		red := canvas.NewPaint()
		red.Color = colorcore.ARGB(255, 255, 0, 0)
		c.DrawRect(geom.RectLTRB(10, 10, 50, 50), red)
		c.Restore()
	})
	validatePDF(t, data)

	// The layer becomes an isolated transparency-group form XObject.
	if !bytes.Contains(data, []byte("/Subtype /Form")) {
		t.Fatal("no form XObject emitted for the layer")
	}
	if !bytes.Contains(data, []byte("/S /Transparency")) {
		t.Error("layer form XObject is not a transparency group (/S /Transparency)")
	}

	// The page content applies the layer alpha /ExtGState and invokes the XObject; the layer geometry lives in the
	// XObject, not the page content.
	content := pageContent(t, data)
	mustContain(t, content, " gs\n")
	mustContain(t, content, " Do\n")
	if strings.Contains(content, " re\n") {
		t.Errorf("layer geometry leaked into the page content:\n%s", content)
	}
	foundAlpha := false
	for _, s := range dictObjects(data) {
		if strings.Contains(s, "/ca .502") { // 128/255 to four significant digits
			foundAlpha = true
		}
	}
	if !foundAlpha {
		t.Error("no /ExtGState with the ~0.5 layer alpha")
	}
}

func TestDeviceSaveLayerAdvancedBlendComposites(t *testing.T) {
	// A non-PDF-expressible layer blend mode (Modulate) over existing content takes the form-XObject compositing path
	// through setUpContentEntry/finishContentEntry.
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		base := canvas.NewPaint()
		base.Color = colorcore.ARGB(255, 0, 0, 255)
		c.DrawRect(geom.RectLTRB(0, 0, 100, 100), base)

		lp := canvas.NewPaint()
		lp.BlendMode = raster.BlendModulate
		c.SaveLayer(nil, lp)
		red := canvas.NewPaint()
		red.Color = colorcore.ARGB(255, 255, 0, 0)
		c.DrawRect(geom.RectLTRB(10, 10, 50, 50), red)
		c.Restore()
	})
	validatePDF(t, data)

	var foundSMask, foundNone, foundMultiply bool
	for _, s := range dictObjects(data) {
		if strings.Contains(s, "/SMask <<") && strings.Contains(s, "/S /Alpha") {
			foundSMask = true
		}
		if strings.Contains(s, "/SMask /None") {
			foundNone = true
		}
		if strings.Contains(s, "/BM /Multiply") {
			foundMultiply = true
		}
	}
	if !foundSMask {
		t.Error("no alpha soft-mask /ExtGState from the advanced-blend compositing")
	}
	if !foundNone {
		t.Error("no /SMask /None (clearMaskOnGraphicState) state emitted")
	}
	if !foundMultiply {
		t.Error("no /BM /Multiply /ExtGState from the Modulate compositing")
	}
	// The dance captures the destination, the source, and the shape mask as separate form XObjects.
	if n := bytes.Count(data, []byte("/Subtype /Form")); n < 3 {
		t.Errorf("expected >=3 form XObjects for the Modulate dance, got %d", n)
	}
}

// dictObjects returns the bodies of all non-stream indirect objects (dictionaries).
func dictObjects(data []byte) []string {
	var out []string
	rest := data
	for {
		oi := bytes.Index(rest, []byte(" obj\n"))
		if oi < 0 {
			break
		}
		ei := bytes.Index(rest[oi:], []byte("\nendobj"))
		if ei < 0 {
			break
		}
		body := string(rest[oi+len(" obj\n") : oi+ei])
		if !strings.Contains(body, " stream\n") {
			out = append(out, body)
		}
		rest = rest[oi+ei+len("\nendobj"):]
	}
	return out
}
