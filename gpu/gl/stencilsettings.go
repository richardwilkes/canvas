// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The abstract description of how a draw uses the stencil buffer (UserStencilSettings), and its concrete translation
// given the stencil-clip state and the target's stencil bit count (StencilSettings). The flag/mask deductions are
// computed by a runtime constructor invoked from package-level variable initializers, preserving pointer identity for
// the shared sentinels (unusedStencilSettings and the clip-bit tables). Program-key generation for these settings is
// trimmed: GL keys no fixed-function state, so only non-GL backends needed it.

package gl

import "github.com/richardwilkes/canvas/gpu"

// UserStencilTest is the abstract stencil test a draw requests.
type UserStencilTest uint16

// UserStencilTest values.
const (
	// Tests that respect the clip bit. If a stencil clip is not in effect, the "IfInClip" is ignored and these only act
	// on user bits.
	UserStencilTestAlwaysIfInClip UserStencilTest = iota
	UserStencilTestEqualIfInClip
	UserStencilTestLessIfInClip
	UserStencilTestLEqualIfInClip

	// Tests that ignore the clip bit. The client is responsible to ensure no color write occurs outside the clip if it
	// is in use.
	UserStencilTestAlways
	UserStencilTestNever
	UserStencilTestGreater
	UserStencilTestGEqual
	UserStencilTestLess
	UserStencilTestLEqual
	UserStencilTestEqual
	UserStencilTestNotEqual
)

// lastClippedStencilTest is the last UserStencilTest value that respects the clip bit.
const lastClippedStencilTest = UserStencilTestLEqualIfInClip

// UserStencilOp is the abstract stencil op a draw requests.
type UserStencilOp uint8

// UserStencilOp values.
const (
	UserStencilOpKeep UserStencilOp = iota

	// Ops that only modify user bits. These must not be paired with ops that modify the clip bit.
	UserStencilOpZero
	UserStencilOpReplace // Replace stencil value with Ref (only the bits enabled in WriteMask).
	UserStencilOpInvert
	UserStencilOpIncWrap
	UserStencilOpDecWrap
	// These two should only be used if wrap ops are not supported, or if the math is guaranteed to not overflow. The
	// user bits may or may not clamp, depending on the state of non-user bits.
	UserStencilOpIncMaybeClamp
	UserStencilOpDecMaybeClamp

	// Ops that only modify the clip bit. These must not be paired with ops that modify user bits.
	UserStencilOpZeroClipBit
	UserStencilOpSetClipBit
	UserStencilOpInvertClipBit

	// Ops that modify both clip and user bits. These can only be paired with kKeep or each other.
	UserStencilOpSetClipAndReplaceUserBits
	UserStencilOpZeroClipAndUserBits
)

// lastUserOnlyStencilOp / lastClipOnlyStencilOp bound the "user bits only" and "clip bit only" op ranges.
const (
	lastUserOnlyStencilOp = UserStencilOpDecMaybeClamp
	lastClipOnlyStencilOp = UserStencilOpInvertClipBit
)

// Stencil flags.
const (
	stencilFlagDisabled         uint16 = 1 << 0
	stencilFlagTestAlwaysPasses uint16 = 1 << 1
	stencilFlagNoModifyStencil  uint16 = 1 << 2
	stencilFlagNoWrapOps        uint16 = 1 << 3
	stencilFlagSingleSided      uint16 = 1 << 4

	stencilFlagsLast uint16 = stencilFlagSingleSided
	stencilFlagsAll  uint16 = stencilFlagsLast | (stencilFlagsLast - 1)
)

// UserStencilFace is the abstract per-face stencil test/op configuration for one winding direction.
type UserStencilFace struct {
	Ref       uint16          // Reference value for stencil test and ops.
	Test      UserStencilTest // Stencil test function, where Ref is on the left side.
	TestMask  uint16          // Bitwise "and" applied to Ref and stencil values before testing.
	PassOp    UserStencilOp   // Op to perform when the test passes.
	FailOp    UserStencilOp   // Op to perform when the test fails.
	WriteMask uint16          // Which bits in the stencil buffer should be updated.
}

// UserStencilSettings is an abstract representation of a draw's stencil usage, later translated into concrete
// StencilSettings when the pipeline is finalized. Instances are shared package-level variables; identity comparison
// against UnusedStencilSettings() tests for the shared "no stencil" sentinel.
type UserStencilSettings struct {
	CWFlags  [2]uint16 // cwFlagsForDraw = CWFlags[hasStencilClip]
	CWFace   UserStencilFace
	CCWFlags [2]uint16
	CCWFace  UserStencilFace
}

// userStencilAttrFlags computes the derived stencil flags for one (test, passOp, failOp) triple.
func userStencilAttrFlags(test UserStencilTest, passOp, failOp UserStencilOp, hasStencilClip bool) uint16 {
	// Ensure an op that only modifies user bits isn't paired with one that modifies clip bits, and that an op that only
	// modifies clip bits isn't paired with one that modifies both.
	if passOp != UserStencilOpKeep && failOp != UserStencilOpKeep {
		if (passOp <= lastUserOnlyStencilOp) != (failOp <= lastUserOnlyStencilOp) ||
			(passOp <= lastClipOnlyStencilOp) != (failOp <= lastClipOnlyStencilOp) {
			panic("invalid user stencil op pairing")
		}
	}
	testAlwaysPasses := (!hasStencilClip && test == UserStencilTestAlwaysIfInClip) ||
		test == UserStencilTestAlways
	doesNotModifyStencil := (test == UserStencilTestNever || passOp == UserStencilOpKeep) &&
		(testAlwaysPasses || failOp == UserStencilOpKeep)
	isDisabled := testAlwaysPasses && doesNotModifyStencil
	usesWrapOps := passOp == UserStencilOpIncWrap || passOp == UserStencilOpDecWrap ||
		failOp == UserStencilOpIncWrap || failOp == UserStencilOpDecWrap

	var flags uint16
	if isDisabled {
		flags |= stencilFlagDisabled
	}
	if testAlwaysPasses {
		flags |= stencilFlagTestAlwaysPasses
	}
	if doesNotModifyStencil {
		flags |= stencilFlagNoModifyStencil
	}
	if !usesWrapOps {
		flags |= stencilFlagNoWrapOps
	}
	return flags
}

// userStencilTestIgnoresRef reports whether test never reads the reference value.
func userStencilTestIgnoresRef(test UserStencilTest) bool {
	return test == UserStencilTestAlwaysIfInClip || test == UserStencilTestAlways ||
		test == UserStencilTestNever
}

// userStencilEffectiveTestMask returns the test mask actually used, zero when the test ignores it.
func userStencilEffectiveTestMask(test UserStencilTest, testMask uint16) uint16 {
	if userStencilTestIgnoresRef(test) {
		return 0
	}
	return testMask
}

// userStencilEffectiveWriteMask returns the write mask actually used: zero when the face never modifies the stencil
// (evaluated with hasStencilClip=true — either the entire face gets disabled when hasStencilClip=false, or the
// effective mask is the same).
func userStencilEffectiveWriteMask(test UserStencilTest, passOp, failOp UserStencilOp, writeMask uint16) uint16 {
	if userStencilAttrFlags(test, passOp, failOp, true)&stencilFlagNoModifyStencil != 0 {
		return 0
	}
	return writeMask
}

// MakeUserStencilSettings builds single-sided stencil settings: both faces share the same settings and the single-sided
// flag is set.
func MakeUserStencilSettings(ref uint16, test UserStencilTest, testMask uint16, passOp, failOp UserStencilOp, writeMask uint16) UserStencilSettings {
	face := UserStencilFace{
		Ref:       ref,
		Test:      test,
		TestMask:  userStencilEffectiveTestMask(test, testMask),
		PassOp:    passOp,
		FailOp:    failOp,
		WriteMask: userStencilEffectiveWriteMask(test, passOp, failOp, writeMask),
	}
	flags := [2]uint16{
		userStencilAttrFlags(test, passOp, failOp, false) | stencilFlagSingleSided,
		userStencilAttrFlags(test, passOp, failOp, true) | stencilFlagSingleSided,
	}
	return UserStencilSettings{CWFlags: flags, CWFace: face, CCWFlags: flags, CCWFace: face}
}

// MakeUserStencilSettingsSeparate builds two-sided stencil settings with independent CW/CCW faces (consumed by the
// path-renderer winding passes).
func MakeUserStencilSettingsSeparate(cw, ccw UserStencilFace) UserStencilSettings {
	return UserStencilSettings{
		CWFlags: [2]uint16{
			userStencilAttrFlags(cw.Test, cw.PassOp, cw.FailOp, false),
			userStencilAttrFlags(cw.Test, cw.PassOp, cw.FailOp, true),
		},
		CWFace: UserStencilFace{
			Ref:       cw.Ref,
			Test:      cw.Test,
			TestMask:  userStencilEffectiveTestMask(cw.Test, cw.TestMask),
			PassOp:    cw.PassOp,
			FailOp:    cw.FailOp,
			WriteMask: userStencilEffectiveWriteMask(cw.Test, cw.PassOp, cw.FailOp, cw.WriteMask),
		},
		CCWFlags: [2]uint16{
			userStencilAttrFlags(ccw.Test, ccw.PassOp, ccw.FailOp, false),
			userStencilAttrFlags(ccw.Test, ccw.PassOp, ccw.FailOp, true),
		},
		CCWFace: UserStencilFace{
			Ref:      ccw.Ref,
			Test:     ccw.Test,
			TestMask: userStencilEffectiveTestMask(ccw.Test, ccw.TestMask),
			PassOp:   ccw.PassOp,
			FailOp:   ccw.FailOp,
			WriteMask: userStencilEffectiveWriteMask(ccw.Test, ccw.PassOp, ccw.FailOp,
				ccw.WriteMask),
		},
	}
}

// Flags returns the combined CW/CCW stencil flags for the given clip state.
func (s *UserStencilSettings) Flags(hasStencilClip bool) uint16 {
	i := 0
	if hasStencilClip {
		i = 1
	}
	return s.CWFlags[i] & s.CCWFlags[i]
}

// IsDisabled reports whether stenciling is disabled for the given clip state.
func (s *UserStencilSettings) IsDisabled(hasStencilClip bool) bool {
	return s.Flags(hasStencilClip)&stencilFlagDisabled != 0
}

// TestAlwaysPasses reports whether the stencil test always passes for the given clip state.
func (s *UserStencilSettings) TestAlwaysPasses(hasStencilClip bool) bool {
	return s.Flags(hasStencilClip)&stencilFlagTestAlwaysPasses != 0
}

// IsTwoSided reports whether CW and CCW faces differ for the given clip state.
func (s *UserStencilSettings) IsTwoSided(hasStencilClip bool) bool {
	return s.Flags(hasStencilClip)&stencilFlagSingleSided == 0
}

// UsesWrapOp reports whether either face uses a wrapping increment/decrement op.
func (s *UserStencilSettings) UsesWrapOp(hasStencilClip bool) bool {
	return s.Flags(hasStencilClip)&stencilFlagNoWrapOps == 0
}

// IsUnused reports whether these are the shared "no stencil" sentinel settings (by pointer identity).
func (s *UserStencilSettings) IsUnused() bool { return s == unusedStencilSettings }

// unusedStencilSettings is the shared sentinel meaning "no stencil work."
var unusedStencilSettings = func() *UserStencilSettings {
	s := MakeUserStencilSettings(0x0000, UserStencilTestAlwaysIfInClip, 0xffff,
		UserStencilOpKeep, UserStencilOpKeep, 0x0000)
	if stencilFlagsAll != s.CWFlags[0]&s.CCWFlags[0] {
		panic("kUnused must carry every stencil flag")
	}
	return &s
}()

// UnusedStencilSettings returns the shared sentinel meaning "no stencil work."
func UnusedStencilSettings() *UserStencilSettings { return unusedStencilSettings }

//////////////////////////////////////////////////////////////////////////////
// StencilSettings: the concrete, hardware-facing translation.

// StencilTest is the concrete, hardware-level stencil test function.
type StencilTest uint16

// StencilTest values.
const (
	StencilTestAlways StencilTest = iota
	StencilTestNever
	StencilTestGreater
	StencilTestGEqual
	StencilTestLess
	StencilTestLEqual
	StencilTestEqual
	StencilTestNotEqual
)

// StencilOp is the concrete, hardware-level stencil op.
type StencilOp uint8

// StencilOp values.
const (
	StencilOpKeep StencilOp = iota
	StencilOpZero
	StencilOpReplace // Replace stencil value with Ref (only the bits enabled in WriteMask).
	StencilOpInvert
	StencilOpIncWrap
	StencilOpDecWrap
	// NOTE: clamping occurs before the write mask. So if the MSB is zero and masked out, stencil values will still wrap
	// when using clamping ops.
	StencilOpIncClamp
	StencilOpDecClamp
)

// stencilFlagInvalidPrivate marks a StencilSettings as not-yet-reset / invalidated.
const stencilFlagInvalidPrivate = uint32(stencilFlagsLast) << 1

// StencilFace is the concrete, hardware-level per-face stencil configuration. The struct is comparable via ==.
type StencilFace struct {
	Ref       uint16
	Test      StencilTest
	TestMask  uint16
	PassOp    StencilOp
	FailOp    StencilOp
	WriteMask uint16
}

// userStencilTestToRaw maps each UserStencilTest to its concrete StencilTest.
var userStencilTestToRaw = [...]StencilTest{
	// Tests that respect the clip.
	StencilTestAlways, // kAlwaysIfInClip (this is only for when there is not a stencil clip).
	StencilTestEqual,  // kEqualIfInClip.
	StencilTestLess,   // kLessIfInClip.
	StencilTestLEqual, // kLEqualIfInClip.

	// Tests that ignore the clip.
	StencilTestAlways,
	StencilTestNever,
	StencilTestGreater,
	StencilTestGEqual,
	StencilTestLess,
	StencilTestLEqual,
	StencilTestEqual,
	StencilTestNotEqual,
}

// userStencilOpToRaw maps each UserStencilOp to its concrete StencilOp.
var userStencilOpToRaw = [...]StencilOp{
	StencilOpKeep,

	// Ops that only modify user bits.
	StencilOpZero,
	StencilOpReplace,
	StencilOpInvert,
	StencilOpIncWrap,
	StencilOpDecWrap,
	StencilOpIncClamp, // kIncMaybeClamp.
	StencilOpDecClamp, // kDecMaybeClamp.

	// Ops that only modify the clip bit.
	StencilOpZero,    // kZeroClipBit.
	StencilOpReplace, // kSetClipBit.
	StencilOpInvert,  // kInvertClipBit.

	// Ops that modify clip and user bits.
	StencilOpReplace, // kSetClipAndReplaceUserBits.
	StencilOpZero,    // kZeroClipAndUserBits.
}

// reset derives the concrete face settings from an abstract UserStencilFace, given the clip state and stencil bit
// count.
func (f *StencilFace) reset(user *UserStencilFace, hasStencilClip bool, numStencilBits int) {
	if numStencilBits <= 0 || numStencilBits > 16 {
		panic("invalid stencil bit count")
	}
	clipBit := uint16(1) << (numStencilBits - 1)
	userMask := clipBit - 1

	maxOp := max(user.PassOp, user.FailOp)
	switch {
	case maxOp <= lastUserOnlyStencilOp:
		// Ops that only modify user bits.
		f.WriteMask = user.WriteMask & userMask
	case maxOp <= lastClipOnlyStencilOp:
		// Ops that only modify the clip bit.
		f.WriteMask = clipBit
	default:
		// Ops that modify both clip and user bits.
		f.WriteMask = clipBit | (user.WriteMask & userMask)
	}

	f.FailOp = userStencilOpToRaw[user.FailOp]
	f.PassOp = userStencilOpToRaw[user.PassOp]

	switch {
	case !hasStencilClip || user.Test > lastClippedStencilTest:
		// Ignore the clip.
		f.TestMask = user.TestMask & userMask
		f.Test = userStencilTestToRaw[user.Test]
	case user.Test != UserStencilTestAlwaysIfInClip:
		// Respect the clip.
		f.TestMask = clipBit | (user.TestMask & userMask)
		f.Test = userStencilTestToRaw[user.Test]
	default:
		// Test only for clip.
		f.TestMask = clipBit
		f.Test = StencilTestEqual
	}

	f.Ref = (clipBit | user.Ref) & (f.TestMask | f.WriteMask)
}

// setDisabled resets the face to the always-pass, no-op zero value.
func (f *StencilFace) setDisabled() {
	*f = StencilFace{}
	if StencilTestAlways != 0 || StencilOpKeep != 0 {
		panic("zero value must mean always/keep")
	}
}

// StencilSettings is the concrete stencil configuration that maps directly to the underlying hardware, deduced from
// user stencil settings, the stencil clip status, and the number of bits in the target stencil buffer.
type StencilSettings struct {
	flags   uint32
	cwFace  StencilFace
	ccwFace StencilFace
}

// MakeStencilSettings derives concrete stencil settings from user, the clip status, and the target's stencil bit count.
func MakeStencilSettings(user *UserStencilSettings, hasStencilClip bool, numStencilBits int) StencilSettings {
	var s StencilSettings
	s.Reset(user, hasStencilClip, numStencilBits)
	return s
}

// Invalidate marks the settings as needing a Reset before use.
func (s *StencilSettings) Invalidate() { s.flags |= stencilFlagInvalidPrivate }

// SetDisabled disables stenciling entirely.
func (s *StencilSettings) SetDisabled() { s.flags = uint32(stencilFlagsAll) }

// Reset (re)derives the concrete settings from user, the clip status, and the target's stencil bit count.
func (s *StencilSettings) Reset(user *UserStencilSettings, hasStencilClip bool, numStencilBits int) {
	clipIdx := 0
	if hasStencilClip {
		clipIdx = 1
	}
	cwFlags := user.CWFlags[clipIdx]
	if cwFlags&stencilFlagSingleSided != 0 {
		if cwFlags != user.CCWFlags[clipIdx] {
			panic("single-sided settings must have matching face flags")
		}
		s.flags = uint32(cwFlags)
		if !s.IsDisabled() {
			s.cwFace.reset(&user.CWFace, hasStencilClip, numStencilBits)
		}
		return
	}

	ccwFlags := user.CCWFlags[clipIdx]
	s.flags = uint32(cwFlags & ccwFlags)
	if s.IsDisabled() {
		return
	}
	if cwFlags&stencilFlagDisabled == 0 {
		s.cwFace.reset(&user.CWFace, hasStencilClip, numStencilBits)
	} else {
		s.cwFace.setDisabled()
	}
	if ccwFlags&stencilFlagDisabled == 0 {
		s.ccwFace.reset(&user.CCWFace, hasStencilClip, numStencilBits)
	} else {
		s.ccwFace.setDisabled()
	}
}

// IsValid reports whether the settings have been derived (via Reset) since the last Invalidate.
func (s *StencilSettings) IsValid() bool { return s.flags&stencilFlagInvalidPrivate == 0 }

// IsDisabled reports whether stenciling is disabled.
func (s *StencilSettings) IsDisabled() bool {
	s.assertValid()
	return s.flags&uint32(stencilFlagDisabled) != 0
}

// DoesWrite reports whether either face writes to the stencil buffer.
func (s *StencilSettings) DoesWrite() bool {
	s.assertValid()
	return s.flags&uint32(stencilFlagNoModifyStencil) == 0
}

// IsTwoSided reports whether the CW and CCW faces differ.
func (s *StencilSettings) IsTwoSided() bool {
	s.assertValid()
	return s.flags&uint32(stencilFlagSingleSided) == 0
}

// UsesWrapOp reports whether either face uses a wrapping increment/decrement op.
func (s *StencilSettings) UsesWrapOp() bool {
	s.assertValid()
	return s.flags&uint32(stencilFlagNoWrapOps) == 0
}

func (s *StencilSettings) assertValid() {
	if !s.IsValid() {
		panic("stencil settings are invalid")
	}
}

// Equal reports whether two stencil settings are equivalent.
func (s *StencilSettings) Equal(that *StencilSettings) bool {
	if (stencilFlagInvalidPrivate|uint32(stencilFlagDisabled))&(s.flags|that.flags) != 0 {
		// At least one is invalid and/or disabled.
		if stencilFlagInvalidPrivate&(s.flags|that.flags) != 0 {
			return false // We never allow invalid stencils to be equal.
		}
		// They're only equal if both are disabled.
		return uint32(stencilFlagDisabled)&(s.flags&that.flags) != 0
	}
	switch {
	case uint32(stencilFlagSingleSided)&(s.flags&that.flags) != 0:
		return s.cwFace == that.cwFace // Both are single sided.
	case uint32(stencilFlagSingleSided)&(s.flags|that.flags) != 0:
		return false
	default:
		return s.cwFace == that.cwFace && s.ccwFace == that.ccwFace
	}
}

// SingleSidedFace returns the shared face configuration for single-sided settings.
func (s *StencilSettings) SingleSidedFace() *StencilFace {
	if s.IsDisabled() || s.IsTwoSided() {
		panic("singleSidedFace requires enabled single-sided settings")
	}
	return &s.cwFace
}

// PostOriginCWFace returns the stencil settings for triangles that wind clockwise in "post-origin" space (the space
// that results after a potential y-axis flip on device space for bottom-left origins).
func (s *StencilSettings) PostOriginCWFace(origin gpu.SurfaceOrigin) *StencilFace {
	if !s.IsTwoSided() {
		panic("postOriginCWFace requires two-sided settings")
	}
	if origin == gpu.OriginTopLeft {
		return &s.cwFace
	}
	return &s.ccwFace
}

// PostOriginCCWFace returns the stencil settings for triangles that wind counterclockwise in "post-origin" space.
func (s *StencilSettings) PostOriginCCWFace(origin gpu.SurfaceOrigin) *StencilFace {
	if !s.IsTwoSided() {
		panic("postOriginCCWFace requires two-sided settings")
	}
	if origin == gpu.OriginTopLeft {
		return &s.ccwFace
	}
	return &s.cwFace
}

// The clip-bit set/zero settings.
var (
	zeroStencilClipBit = MakeUserStencilSettings(0x0000, UserStencilTestAlways, 0xffff,
		UserStencilOpZeroClipBit, UserStencilOpZeroClipBit, 0x0000)
	setStencilClipBit = MakeUserStencilSettings(0x0000, UserStencilTestAlways, 0xffff,
		UserStencilOpSetClipBit, UserStencilOpSetClipBit, 0x0000)
)

// SetClipBitSettings returns the shared stencil settings that set (or zero) the clip bit.
func SetClipBitSettings(setToInside bool) *UserStencilSettings {
	if setToInside {
		return &setStencilClipBit
	}
	return &zeroStencilClipBit
}
