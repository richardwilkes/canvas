// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The fragment-processor (FP) tree model and its program-impl counterpart: child registration with sample usage, the
// optimization/private flag machinery, key generation, program-impl tree construction, and the
// invokeChild/invokeChildWithMatrix code emission helpers. Each concrete FP is a Go interface value with an embedded
// FPBase, and children are plain (GC-owned) interface values. Emitted code is GLSL, so a missing input color defaults
// to the literal "vec4(1.0)".

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
)

// FP optimization flags.
const (
	// FPCompatibleWithCoverageAsAlpha: output is a modulation of the input color or alpha with a computed premultiplied
	// color or alpha that is in the 0..1 range.
	FPCompatibleWithCoverageAsAlpha uint32 = 0x1
	// FPPreservesOpaqueInput: all opaque input colors to the processor produce opaque output.
	FPPreservesOpaqueInput uint32 = 0x2
	// FPConstantOutputForConstantInput: a constant input produces a constant output.
	FPConstantOutputForConstantInput uint32 = 0x4
	// FPAllOptimizationFlags is the union of every optimization flag.
	FPAllOptimizationFlags = FPCompatibleWithCoverageAsAlpha | FPPreservesOpaqueInput |
		FPConstantOutputForConstantInput

	// Private flags, packed above the optimization flags.
	fpUsesSampleCoordsIndirectly = FPAllOptimizationFlags + 1
	fpUsesSampleCoordsDirectly   = fpUsesSampleCoordsIndirectly << 1
	fpIsBlendFunction            = fpUsesSampleCoordsIndirectly << 2
	fpWillReadDstColor           = fpUsesSampleCoordsIndirectly << 3
)

// FragmentProcessor is the interface every fragment processor implements. Concrete FPs embed FPBase and implement the
// on-hooks; the tree-walking logic lives in free functions that take the interface.
type FragmentProcessor interface {
	Processor

	// Clone returns a deep copy of the FP, including a clone of each child.
	Clone() FragmentProcessor

	// fpBase gives shared code access to the embedded base.
	fpBase() *FPBase

	// onMakeProgramImpl builds the ProgramImpl that emits this FP's shader code.
	onMakeProgramImpl() FPProgramImpl
	// onAddToKey appends this FP's subclass-specific key bytes to b.
	onAddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder)
	// onIsEqual reports whether other (same ClassID) is equivalent to this FP; only called when ClassIDs match.
	onIsEqual(other FragmentProcessor) bool
	// constantOutputForConstantInput computes the FP's output for a constant input; only legal when the
	// FPConstantOutputForConstantInput flag is set.
	constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f
}

// FPBase holds the fields shared by every fragment processor: its parent/child links, sample usage, and
// optimization/private flags.
type FPBase struct {
	parent   FragmentProcessor
	children []FragmentProcessor
	flags    uint32
	usage    SampleUsage
	processorBase
}

// initFP initializes the shared FP fields; every concrete FP constructor calls this first.
func (b *FPBase) initFP(classID ClassID, optimizationFlags uint32) {
	if optimizationFlags&^FPAllOptimizationFlags != 0 {
		panic("invalid optimization flags")
	}
	b.classID = classID
	b.flags = optimizationFlags
}

func (b *FPBase) fpBase() *FPBase { return b }

// constantOutputForConstantInput is the base "must override" default.
func (b *FPBase) constantOutputForConstantInput(colorcore.PMColor4f) colorcore.PMColor4f {
	panic("subclass must override constantOutputForConstantInput to advertise this optimization")
}

// NumChildProcessors returns the number of registered child FP slots (including nil placeholders).
func (b *FPBase) NumChildProcessors() int { return len(b.children) }

// ChildProcessor returns the child FP at index (may be nil for an unfilled slot).
func (b *FPBase) ChildProcessor(index int) FragmentProcessor { return b.children[index] }

// Parent returns the FP that registered this one as a child, or nil for a root FP.
func (b *FPBase) Parent() FragmentProcessor { return b.parent }

// SampleUsage reports how this FP is invoked by its parent.
func (b *FPBase) SampleUsage() SampleUsage { return b.usage }

// WillReadDstColor reports whether this FP (or any of its children) reads the destination color.
func (b *FPBase) WillReadDstColor() bool { return b.flags&fpWillReadDstColor != 0 }

// IsBlendFunction reports whether this FP takes a destination color as a second input.
func (b *FPBase) IsBlendFunction() bool { return b.flags&fpIsBlendFunction != 0 }

// UsesSampleCoordsDirectly reports whether this FP itself (not a descendant) reads its sample coordinates.
func (b *FPBase) UsesSampleCoordsDirectly() bool { return b.flags&fpUsesSampleCoordsDirectly != 0 }

// UsesSampleCoords reports whether this FP or any descendant reads sample coordinates.
func (b *FPBase) UsesSampleCoords() bool {
	return b.flags&(fpUsesSampleCoordsDirectly|fpUsesSampleCoordsIndirectly) != 0
}

// CompatibleWithCoverageAsAlpha reports whether this FP's output can have coverage folded into its alpha channel.
func (b *FPBase) CompatibleWithCoverageAsAlpha() bool {
	return b.flags&FPCompatibleWithCoverageAsAlpha != 0
}

// PreservesOpaqueInput reports whether an opaque input to this FP always produces opaque output.
func (b *FPBase) PreservesOpaqueInput() bool { return b.flags&FPPreservesOpaqueInput != 0 }

// HasConstantOutputForConstantInput reports whether constantOutputForConstantInput may be called on this FP.
func (b *FPBase) HasConstantOutputForConstantInput() bool {
	return b.flags&FPConstantOutputForConstantInput != 0
}

// ClearConstantOutputForConstantInputFlag drops the FPConstantOutputForConstantInput flag, e.g. once a caller has
// consumed the constant-folded value and the FP is retained for other reasons.
func (b *FPBase) ClearConstantOutputForConstantInputFlag() {
	b.flags &^= FPConstantOutputForConstantInput
}

// optimizationFlags returns the FP's public optimization flags, masking off the private bits.
func (b *FPBase) optimizationFlags() uint32 { return b.flags & FPAllOptimizationFlags }

// setUsesSampleCoordsDirectly marks this FP as reading its own sample coordinates.
func (b *FPBase) setUsesSampleCoordsDirectly() { b.flags |= fpUsesSampleCoordsDirectly }

// setWillReadDstColor marks this FP as reading the destination color.
func (b *FPBase) setWillReadDstColor() { b.flags |= fpWillReadDstColor }

// setIsBlendFunction marks this FP as taking a destination color as a second input.
func (b *FPBase) setIsBlendFunction() { b.flags |= fpIsBlendFunction }

// mergeOptimizationFlags narrows this FP's optimization flags to the intersection with flags, leaving the private bits
// untouched.
func (b *FPBase) mergeOptimizationFlags(flags uint32) {
	if flags&^FPAllOptimizationFlags != 0 {
		panic("invalid optimization flags")
	}
	b.flags &= flags | ^FPAllOptimizationFlags
}

// registerChild attaches a child FP with the given sample usage, propagating relevant flags up to the owner. It must be
// called AFTER the owner's initFP. A nil child records a placeholder slot.
func (b *FPBase) registerChild(child FragmentProcessor, sampleUsage SampleUsage) {
	if !sampleUsage.IsSampled() {
		panic("child must be sampled")
	}
	if child == nil {
		b.children = append(b.children, nil)
		return
	}
	cb := child.fpBase()
	if cb.parent != nil || cb.usage.IsSampled() {
		panic("child already attached to another FP")
	}

	// Configure child's sampling state first.
	cb.usage = sampleUsage

	// Propagate the "will read dest-color" flag up to parent FPs.
	if cb.WillReadDstColor() {
		b.setWillReadDstColor()
	}

	// If this child receives passthrough or matrix transformed coords from its parent then note that the parent's
	// coords are used indirectly to ensure that they aren't omitted.
	if (sampleUsage.IsPassThrough() || sampleUsage.IsUniformMatrix()) && cb.UsesSampleCoords() {
		b.flags |= fpUsesSampleCoordsIndirectly
	}

	// The child's parent pointer is set to the owner by registerChildOf (Go cannot recover the owning interface value
	// from the embedded base). See registerChildOf.
	b.children = append(b.children, child)
}

// registerChildOf is the form of registerChild that also records the parent back-pointer; owner must be the FP whose
// base holds the child.
func registerChildOf(owner, child FragmentProcessor, sampleUsage SampleUsage) {
	owner.fpBase().registerChild(child, sampleUsage)
	if child != nil {
		child.fpBase().parent = owner
	}
}

// cloneAndRegisterAllChildProcessors clones every child of src and registers the clones on dst with the same sample
// usages.
func cloneAndRegisterAllChildProcessors(dst, src FragmentProcessor) {
	sb := src.fpBase()
	for i := 0; i < sb.NumChildProcessors(); i++ {
		if child := sb.ChildProcessor(i); child != nil {
			registerChildOf(dst, child.Clone(), child.fpBase().SampleUsage())
		} else {
			dst.fpBase().registerChild(nil, PassThroughSampleUsage())
		}
	}
}

// fpProcessorOptimizationFlags returns fp's optimization flags, or FPAllOptimizationFlags if fp is nil (an absent child
// imposes no constraint).
func fpProcessorOptimizationFlags(fp FragmentProcessor) uint32 {
	if fp != nil {
		return fp.fpBase().optimizationFlags()
	}
	return FPAllOptimizationFlags
}

// fpConstantOutputForConstantInput returns fp's constant output for input, or input unchanged if fp is nil. Panics if
// fp doesn't advertise the optimization.
func fpConstantOutputForConstantInput(fp FragmentProcessor, input colorcore.PMColor4f) colorcore.PMColor4f {
	if fp == nil {
		return input
	}
	if !fp.fpBase().HasConstantOutputForConstantInput() {
		panic("FP does not advertise constant output for constant input")
	}
	return fp.constantOutputForConstantInput(input)
}

// fpHasConstantOutputForConstantInput returns fp's constant output for input and true if fp advertises the
// optimization, or the zero color and false otherwise.
func fpHasConstantOutputForConstantInput(fp FragmentProcessor, input colorcore.PMColor4f) (colorcore.PMColor4f, bool) {
	if fp.fpBase().HasConstantOutputForConstantInput() {
		return fp.constantOutputForConstantInput(input), true
	}
	return colorcore.PMColor4f{}, false
}

// fpAddToKey appends fp's subclass key followed by its children's keys, recursively.
func fpAddToKey(fp FragmentProcessor, caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	fp.onAddToKey(caps, b)
	for _, child := range fp.fpBase().children {
		if child != nil {
			fpAddToKey(child, caps, b)
		}
	}
}

// fpIsEqual reports whether a and b are equivalent FP trees: same class, same sample usage, equal subclass state, and
// equal children.
func fpIsEqual(a, b FragmentProcessor) bool {
	if a.ClassID() != b.ClassID() {
		return false
	}
	if a.fpBase().SampleUsage() != b.fpBase().SampleUsage() {
		return false
	}
	if !a.onIsEqual(b) {
		return false
	}
	if a.fpBase().NumChildProcessors() != b.fpBase().NumChildProcessors() {
		return false
	}
	for i := 0; i < a.fpBase().NumChildProcessors(); i++ {
		ac := a.fpBase().ChildProcessor(i)
		bc := b.fpBase().ChildProcessor(i)
		if (ac != nil) != (bc != nil) {
			return false
		}
		if ac != nil && !fpIsEqual(ac, bc) {
			return false
		}
	}
	return true
}

// fpVisitTextureEffects calls f for every TextureEffect in fp's tree.
func fpVisitTextureEffects(fp FragmentProcessor, f func(*TextureEffect)) {
	if te, ok := fp.(*TextureEffect); ok {
		f(te)
	}
	for _, child := range fp.fpBase().children {
		if child != nil {
			fpVisitTextureEffects(child, f)
		}
	}
}

// fpVisitProxies calls f for every surface proxy sampled by fp's tree.
func fpVisitProxies(fp FragmentProcessor, f func(*SurfaceProxy, gpu.Mipmapped)) {
	fpVisitTextureEffects(fp, func(te *TextureEffect) {
		f(te.view.Proxy(), te.view.Mipmapped())
	})
}

// makeFPProgramImpl builds the program-impl tree matching fp's FP tree.
func makeFPProgramImpl(fp FragmentProcessor) FPProgramImpl {
	impl := fp.onMakeProgramImpl()
	base := impl.fpImplBase()
	base.children = make([]FPProgramImpl, len(fp.fpBase().children))
	for i, child := range fp.fpBase().children {
		if child != nil {
			base.children[i] = makeFPProgramImpl(child)
		}
	}
	return impl
}

// fpVisitWithImpls walks fp and impl in lockstep, calling f for each corresponding pair.
func fpVisitWithImpls(fp FragmentProcessor, impl FPProgramImpl, f func(FragmentProcessor, FPProgramImpl)) {
	f(fp, impl)
	if impl.fpImplBase().NumChildProcessors() != fp.fpBase().NumChildProcessors() {
		panic("impl tree does not match FP tree")
	}
	for i := 0; i < fp.fpBase().NumChildProcessors(); i++ {
		if child := fp.fpBase().ChildProcessor(i); child != nil {
			fpVisitWithImpls(child, impl.fpImplBase().ChildProcessor(i), f)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////

// FPProgramImpl is the program-building counterpart of a FragmentProcessor: it emits GLSL code and uploads uniform
// values for one FP.
type FPProgramImpl interface {
	// EmitCode emits this FP's GLSL fragment-shader code.
	EmitCode(args *FPEmitArgs)
	// onSetData uploads this FP's uniform values for the given FP instance.
	onSetData(pdman *ProgramDataManager, fp FragmentProcessor)
	// fpImplBase gives shared code access to the embedded base.
	fpImplBase() *FPImplBase
}

// FPImplBase holds the state common to every FPProgramImpl: its emitted function name and its children's impls, plus
// the default no-op onSetData.
type FPImplBase struct {
	functionName string
	children     []FPProgramImpl
}

func (b *FPImplBase) fpImplBase() *FPImplBase { return b }

// onSetData is the default no-op.
func (b *FPImplBase) onSetData(*ProgramDataManager, FragmentProcessor) {}

// NumChildProcessors returns the number of child impls.
func (b *FPImplBase) NumChildProcessors() int { return len(b.children) }

// ChildProcessor returns the child impl at index.
func (b *FPImplBase) ChildProcessor(index int) FPProgramImpl { return b.children[index] }

// SetFunctionName records the GLSL function name emitted for this FP. May only be called once.
func (b *FPImplBase) SetFunctionName(name string) {
	if b.functionName != "" {
		panic("function name already set")
	}
	b.functionName = name
}

// FunctionName returns the GLSL function name previously recorded by SetFunctionName.
func (b *FPImplBase) FunctionName() string {
	if b.functionName == "" {
		panic("function name not set")
	}
	return b.functionName
}

// fpImplSetData uploads impl's uniform values for the given FP instance (no recursion; callers walk the tree).
func fpImplSetData(impl FPProgramImpl, pdman *ProgramDataManager, fp FragmentProcessor) {
	impl.onSetData(pdman, fp)
}

// FPEmitArgs bundles the state EmitCode needs to emit one FP's GLSL. InputColor is never empty; a missing input color
// is represented as the literal "vec4(1.0)".
type FPEmitArgs struct {
	FragBuilder    *FragmentShaderBuilder
	UniformHandler *UniformHandler
	ShaderCaps     *gpu.ShaderCaps
	FP             FragmentProcessor
	InputColor     string
	DestColor      string
	SampleCoord    string
}

func makeFPEmitArgs(fragBuilder *FragmentShaderBuilder, uniformHandler *UniformHandler, caps *gpu.ShaderCaps, fp FragmentProcessor, inputColor, destColor, sampleCoord string) *FPEmitArgs {
	if inputColor == "" {
		inputColor = "vec4(1.0)"
	}
	return &FPEmitArgs{
		FragBuilder:    fragBuilder,
		UniformHandler: uniformHandler,
		ShaderCaps:     caps,
		FP:             fp,
		InputColor:     inputColor,
		DestColor:      destColor,
		SampleCoord:    sampleCoord,
	}
}

// InvokeChild emits a call to the given child with the default dest color and parent coords.
func (b *FPImplBase) InvokeChild(childIndex int, args *FPEmitArgs) string {
	return b.InvokeChildFull(childIndex, "", "", args, "")
}

// InvokeChildWithColor emits a call to the given child with an explicit input color.
func (b *FPImplBase) InvokeChildWithColor(childIndex int, inputColor string, args *FPEmitArgs) string {
	return b.InvokeChildFull(childIndex, inputColor, "", args, "")
}

// InvokeChildWithCoords emits a call to the given child with an explicit input color and sample coordinate expression.
func (b *FPImplBase) InvokeChildWithCoords(childIndex int, inputColor string, args *FPEmitArgs, coords string) string {
	return b.InvokeChildFull(childIndex, inputColor, "", args, coords)
}

// InvokeChildFull emits a call to the given child with explicit input color, dest color, and sample coordinate
// expression. Empty strings mean "use the default for this parameter".
func (b *FPImplBase) InvokeChildFull(childIndex int, inputColor, destColor string, args *FPEmitArgs, coords string) string {
	if childIndex < 0 {
		panic("negative child index")
	}
	if inputColor == "" {
		inputColor = args.InputColor
	}

	childProc := args.FP.fpBase().ChildProcessor(childIndex)
	if childProc == nil {
		// If no child processor is provided, return the input color as-is.
		return inputColor
	}

	invocation := fmt.Sprintf("%s(%s", b.ChildProcessor(childIndex).fpImplBase().FunctionName(),
		inputColor)

	if childProc.fpBase().IsBlendFunction() {
		if destColor == "" {
			if args.FP.fpBase().IsBlendFunction() {
				destColor = args.DestColor
			} else {
				destColor = glslOpaqueWhite
			}
		}
		invocation += ", " + destColor
	}

	// A uniform-matrix sample call would go through InvokeChildWithMatrix, not here.
	if childProc.fpBase().SampleUsage().IsUniformMatrix() {
		panic("uniform-matrix sampled child must use InvokeChildWithMatrix")
	}

	if args.FragBuilder.programBuilder.fragmentProcessorHasCoordsParam(childProc) {
		if childProc.fpBase().SampleUsage().IsFragCoord() {
			if coords != fragCoordName {
				panic("frag-coord sampled child must pass " + fragCoordName)
			}
			coords = args.FragBuilder.FragmentPosition()
		}
		if coords != "" {
			invocation += ", " + coords
		} else {
			invocation += ", " + args.SampleCoord
		}
	}

	invocation += ")"
	return invocation
}

// InvokeChildWithMatrix emits a call to a uniform-matrix-sampled child with the default input and dest color.
func (b *FPImplBase) InvokeChildWithMatrix(childIndex int, args *FPEmitArgs) string {
	return b.InvokeChildWithMatrixColor(childIndex, "", args)
}

// InvokeChildWithMatrixColor emits a call to a uniform-matrix-sampled child with an explicit input color.
func (b *FPImplBase) InvokeChildWithMatrixColor(childIndex int, inputColor string, args *FPEmitArgs) string {
	if childIndex < 0 {
		panic("negative child index")
	}
	if inputColor == "" {
		inputColor = args.InputColor
	}

	childProc := args.FP.fpBase().ChildProcessor(childIndex)
	if childProc == nil {
		return inputColor
	}
	if !childProc.fpBase().SampleUsage().IsUniformMatrix() {
		panic("InvokeChildWithMatrix requires a uniform-matrix sampled child")
	}

	// Every uniform matrix has the same (initial) name. Resolve that into the mangled name.
	uniform := args.UniformHandler.getUniformMapping(args.FP, MatrixUniformName())
	if uniform.Type() != GLSLTypeFloat3x3 {
		panic("sample matrix uniform must be a float3x3")
	}
	matrixName := uniform.Name()

	invocation := fmt.Sprintf("%s(%s", b.ChildProcessor(childIndex).fpImplBase().FunctionName(),
		inputColor)

	if childProc.fpBase().IsBlendFunction() {
		destColor := glslOpaqueWhite
		if args.FP.fpBase().IsBlendFunction() {
			destColor = args.DestColor
		}
		invocation += ", " + destColor
	}

	// If the parent coords were produced by uniform transforms, the entire expression (matrixName * coords) is lifted
	// to the vertex shader and stored in a varying; the child will not be sampled explicitly and its function signature
	// will not take coords.
	if args.FragBuilder.programBuilder.fragmentProcessorHasCoordsParam(childProc) {
		// Only check perspective for this specific matrix transform, not the aggregate FP property: any parent
		// perspective has already been applied when evaluated in the FS.
		switch {
		case childProc.fpBase().SampleUsage().HasPerspective():
			invocation += fmt.Sprintf(", proj((%s) * vec3(%s, 1.0))", matrixName, args.SampleCoord)
			args.FragBuilder.ensureProjFunction()
		case args.ShaderCaps.NonsquareMatrixSupport:
			invocation += fmt.Sprintf(", mat3x2(%s) * vec3(%s, 1.0)", matrixName, args.SampleCoord)
		default:
			invocation += fmt.Sprintf(", ((%s) * vec3(%s, 1.0)).xy", matrixName, args.SampleCoord)
		}
	}

	invocation += ")"
	return invocation
}
