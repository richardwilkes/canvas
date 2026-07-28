// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ProgramBuilder emits the vertex and fragment shaders for a ProgramInfo (the geometry processor first, then
// dst-texture setup, the root fragment processors as functions, then the transfer processor), compiles and links them,
// resolves attribute and uniform locations, and produces the Program. The builder writes GLSL text directly and hands
// it straight to the driver to compile. Program caching is in-memory only (an LRU keyed by ProgramDesc); there is no
// persistent on-disk cache or precompilation lane.

package gl

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/richardwilkes/canvas/gpu"
)

// rtAdjustName is the uniform name for the RT-adjust vector (see rtAdjustVector in glprogram.go).
const rtAdjustName = "u_rtAdjust"

// ProgramBuilder is the merged program builder.
type ProgramBuilder struct {
	gpImpl         GPProgramImpl
	xpImpl         XPProgramImpl
	gpu            *Gpu
	vs             *VertexShaderBuilder
	fs             *FragmentShaderBuilder
	varyingHandler *VaryingHandler
	uniformHandler *UniformHandler
	desc           *ProgramDesc
	programInfo    *ProgramInfo
	fpCoordsMap    FPCoordsMap
	fpImpls        []FPProgramImpl
	// Attribute layout gathered for the Program.
	attributes           []ProgramAttribute
	substageIndices      []int
	localCoordsVar       ShaderVar
	vertexAttributeCnt   int
	instanceAttributeCnt int
	vertexStride         int
	instanceStride       int
	numFragmentSamplers  int
	// Each root processor has a stage index: the GP is stage 0, the first root FP stage 1, etc. Names are mangled by
	// appending _S<stage-index>; child FPs append _c<substage> pieces.
	stageIndex              int
	uniformHandles          BuiltinUniformHandles
	dstTextureSamplerHandle SamplerHandle
	dstTextureOrigin        gpu.SurfaceOrigin
}

func newProgramBuilder(g *Gpu, desc *ProgramDesc, programInfo *ProgramInfo) *ProgramBuilder {
	pb := &ProgramBuilder{
		gpu:         g,
		desc:        desc,
		programInfo: programInfo,
		stageIndex:  -1,
	}
	pb.uniformHandler = newUniformHandler(pb)
	pb.varyingHandler = newVaryingHandler(pb)
	pb.vs = newVertexShaderBuilder(pb)
	pb.fs = newFragmentShaderBuilder(pb)
	return pb
}

// shaderCaps returns the driver's shader capabilities.
func (pb *ProgramBuilder) shaderCaps() *gpu.ShaderCaps { return pb.gpu.Caps().ShaderCaps }

// pipeline returns the program's pipeline (the fragment processor tree and transfer processor).
func (pb *ProgramBuilder) pipeline() *Pipeline { return pb.programInfo.Pipeline() }

// geometryProcessor returns the program's geometry processor.
func (pb *ProgramBuilder) geometryProcessor() GeometryProcessor { return pb.programInfo.GeomProc() }

// snapVerticesToPixelCenters reports whether the pipeline requests vertex positions be snapped to pixel centers.
func (pb *ProgramBuilder) snapVerticesToPixelCenters() bool {
	return pb.pipeline().SnapVerticesToPixelCenters()
}

// hasPointSize reports whether the draw's primitive type requires setting gl_PointSize.
func (pb *ProgramBuilder) hasPointSize() bool {
	return pb.programInfo.PrimitiveType() == gpu.PrimitiveTypePoints
}

// finalizeFragmentSecondaryColor is a hook for backend-specific secondary-output customization; GL needs none.
func (pb *ProgramBuilder) finalizeFragmentSecondaryColor(*ShaderVar) {}

// getMangleSuffix returns the stage-mangling suffix for the current processor (appended to variable, function, and
// symbol names to keep them unique across the whole shader).
func (pb *ProgramBuilder) getMangleSuffix() string {
	if pb.stageIndex < 0 {
		panic("mangling before the first stage")
	}
	suffix := fmt.Sprintf("_S%d", pb.stageIndex)
	for _, c := range pb.substageIndices {
		suffix += fmt.Sprintf("_c%d", c)
	}
	return suffix
}

// nameVariable prefixes (unless prefix is 0) and stage-mangles (unless asked not to) a variable, function, or symbol
// name.
func (pb *ProgramBuilder) nameVariable(prefix byte, name string, mangle bool) string {
	out := name
	if prefix != 0 {
		out = string(prefix) + name
	}
	if mangle {
		// Names containing "__" are reserved; add "x" if needed to avoid consecutive underscores.
		if out != "" && out[len(out)-1] == '_' {
			out += "x"
		}
		out += pb.getMangleSuffix()
	}
	return out
}

// nameExpression assigns *output a fresh mangled variable name derived from baseName, if it doesn't already have one.
func (pb *ProgramBuilder) nameExpression(output *string, baseName string) {
	if *output == "" {
		*output = pb.nameVariable(0, baseName, true)
	}
}

// advanceStage moves to the next processor's stage, opening a fresh fragment-shader code section.
func (pb *ProgramBuilder) advanceStage() {
	pb.stageIndex++
	pb.fs.nextStage()
}

// appendUniformDecls appends the declarations for uniforms visible to the given shader stage(s).
func (pb *ProgramBuilder) appendUniformDecls(visibility ShaderFlags, out *[]byte) {
	pb.uniformHandler.appendUniformDecls(visibility, out)
}

// addRTFlipUniform registers the fragment-shader RT-flip uniform under name.
func (pb *ProgramBuilder) addRTFlipUniform(name string) {
	if pb.uniformHandles.RTFlipUni.IsValid() {
		panic("RT flip uniform already added")
	}
	pb.uniformHandles.RTFlipUni, _ = pb.uniformHandler.internalAddUniformArray(nil,
		ShaderFlagFragment, GLSLTypeFloat2, name, false, 0)
}

// fragmentProcessorHasCoordsParam reports whether fp's emitted function takes an explicit sample coordinates parameter
// (rather than having its coordinates lifted to a varying).
func (pb *ProgramBuilder) fragmentProcessorHasCoordsParam(fp FragmentProcessor) bool {
	if coords, ok := pb.fpCoordsMap[fp]; ok {
		return coords.HasCoordsParam
	}
	return fp.fpBase().UsesSampleCoords()
}

// emitAndInstallProcs emits the geometry processor, dst-texture setup, fragment processor tree, and transfer processor,
// in that order, then emits the vertex-shader coordinate-transform code and validates the total sampler count.
func (pb *ProgramBuilder) emitAndInstallProcs() bool {
	var inputColor, inputCoverage string
	if !pb.emitAndInstallPrimProc(&inputColor, &inputCoverage) {
		return false
	}
	if !pb.emitAndInstallDstTexture() {
		return false
	}
	if !pb.emitAndInstallFragProcs(&inputColor, &inputCoverage) {
		return false
	}
	if !pb.emitAndInstallXferProc(inputColor, inputCoverage) {
		return false
	}
	pb.gpImpl.gpImplBase().emitTransformCode(pb.vs, pb.uniformHandler)
	return pb.checkSamplerCounts()
}

// emitAndInstallPrimProc builds and emits the geometry processor's shader code: its texture samplers, the RT-adjustment
// uniform, and its vertex/fragment code via ProgramImpl.
func (pb *ProgramBuilder) emitAndInstallPrimProc(outputColor, outputCoverage *string) bool {
	geomProc := pb.geometryProcessor()

	// Program builders have a bit of state we need to clear with each effect.
	pb.advanceStage()
	pb.nameExpression(outputColor, "outputColor")
	pb.nameExpression(outputCoverage, "outputCoverage")

	if pb.uniformHandles.RTAdjustmentUni.IsValid() {
		panic("RT adjustment uniform already added")
	}
	pb.uniformHandles.RTAdjustmentUni, _ = pb.uniformHandler.internalAddUniformArray(nil,
		ShaderFlagVertex, GLSLTypeFloat4, rtAdjustName, false, 0)

	if pb.gpImpl != nil {
		panic("GP impl already created")
	}
	pb.gpImpl = geomProc.MakeProgramImpl(pb.shaderCaps())

	texSamplers := make([]SamplerHandle, geomProc.gpBase().NumTextureSamplers())
	for i := range texSamplers {
		ts := geomProc.gpBase().TextureSampler(i)
		texSamplers[i] = pb.emitSampler(ts.TextureType(), ts.Swizzle(),
			fmt.Sprintf("TextureSampler_%d", i))
		if !texSamplers[i].IsValid() {
			return false
		}
	}

	args := &GPEmitArgs{
		VertBuilder:    pb.vs,
		FragBuilder:    pb.fs,
		VaryingHandler: pb.varyingHandler,
		UniformHandler: pb.uniformHandler,
		ShaderCaps:     pb.shaderCaps(),
		GeomProc:       geomProc,
		OutputColor:    *outputColor,
		OutputCoverage: *outputCoverage,
		TexSamplers:    texSamplers,
	}
	pb.fpCoordsMap, pb.localCoordsVar = gpImplEmitCode(pb.gpImpl, args, pb.pipeline())

	return true
}

// emitAndInstallDstTexture emits, when the pipeline reads the destination texture, the sampler setup and
// fragment-shader code that samples it into the _dstColor variable.
func (pb *ProgramBuilder) emitAndInstallDstTexture() bool {
	pb.dstTextureOrigin = gpu.OriginTopLeft

	dstView := pb.pipeline().DstProxyView()
	if pb.pipeline().UsesDstTexture() {
		// Set up a sampler handle for the destination texture.
		dstTextureProxy := dstView.AsTextureProxy()
		if dstTextureProxy == nil {
			panic("dst texture read requires a texture proxy")
		}
		swizzle := dstView.Swizzle()
		handle := pb.emitSampler(dstTextureProxy.Proxy().TextureType(), swizzle,
			"DstTextureSampler")
		if !handle.IsValid() {
			return false
		}
		pb.dstTextureSamplerHandle = handle
		pb.dstTextureOrigin = dstView.Origin()

		// Declare a _dstColor global which samples the dest texture at the top of the FS.
		var dstTextureCoordsName string
		pb.uniformHandles.DstTextureCoordsUni, dstTextureCoordsName = pb.uniformHandler.AddUniform(nil, ShaderFlagFragment, GLSLTypeHalf4,
			"DstTextureCoords")
		if dstTextureProxy.Proxy().TextureType() == gpu.TextureType2D {
			pb.fs.CodeAppendf("vec2 _dstTexCoord = (%s - %s.xy) * %s.zw;\n",
				pb.fs.FragmentPosition(), dstTextureCoordsName, dstTextureCoordsName)
			if pb.dstTextureOrigin == gpu.OriginBottomLeft {
				pb.fs.CodeAppend("_dstTexCoord.y = 1.0 - _dstTexCoord.y;\n")
			}
		} else {
			pb.fs.CodeAppendf("vec2 _dstTexCoord = %s - %s.xy;\n",
				pb.fs.FragmentPosition(), dstTextureCoordsName)
			if pb.dstTextureOrigin == gpu.OriginBottomLeft {
				// For rectangle textures the height is stored in z so the coord can be flipped.
				pb.fs.CodeAppendf("_dstTexCoord.y = %s.z - _dstTexCoord.y;\n",
					dstTextureCoordsName)
			}
		}
		dstColor := pb.fs.DstColor()
		pb.fs.DefinitionAppend("vec4 " + dstColor + ";")
		pb.fs.CodeAppendf("%s = %s;\n", dstColor,
			pb.fs.AppendTextureLookup(pb.uniformHandler, pb.dstTextureSamplerHandle,
				"_dstTexCoord"))
	}
	return true
}

// emitAndInstallFragProcs emits every root fragment processor in the pipeline as a chained function, threading color or
// coverage through each in turn.
func (pb *ProgramBuilder) emitAndInstallFragProcs(color, coverage *string) bool {
	fpCount := pb.pipeline().NumFragmentProcessors()
	if len(pb.fpImpls) != 0 {
		panic("FP impls already created")
	}
	for i := 0; i < fpCount; i++ {
		inOut := coverage
		if pb.pipeline().IsColorFragmentProcessor(i) {
			inOut = color
		}
		fp := pb.pipeline().FragmentProcessor(i)
		impl := makeFPProgramImpl(fp)
		pb.fpImpls = append(pb.fpImpls, impl)
		output := pb.emitRootFragProc(fp, impl, *inOut)
		if output == "" {
			return false
		}
		*inOut = output
	}
	return true
}

// emitRootFragProc emits a single root fragment-processor tree as a function and returns the name of the local variable
// holding its output color.
func (pb *ProgramBuilder) emitRootFragProc(fp FragmentProcessor, impl FPProgramImpl, input string) string {
	if input == "" {
		panic("root FP requires an input expression")
	}

	// Program builders have a bit of state we need to clear with each effect.
	pb.advanceStage()
	var output string
	pb.nameExpression(&output, "output")
	pb.fs.CodeAppendf("vec4 %s;", output)

	samplerIndex := 0
	if !pb.emitTextureSamplersForFPs(fp, impl, &samplerIndex) {
		return ""
	}

	pb.writeFPFunction(fp, impl)

	pb.fs.CodeAppendf("%s = %s;", output,
		pb.invokeFP(fp, impl, input, glslOpaqueWhite, pb.localCoordsVar.Name()))

	return output
}

// emitTextureSamplersForFPs walks the fragment-processor tree with its impls and emits a sampler uniform for each
// texture effect, handing the handle to the effect's impl.
func (pb *ProgramBuilder) emitTextureSamplersForFPs(fp FragmentProcessor, impl FPProgramImpl, samplerIndex *int) bool {
	ok := true
	fpVisitWithImpls(fp, impl, func(vfp FragmentProcessor, vimpl FPProgramImpl) {
		if te, isTE := vfp.(*TextureEffect); isTE {
			name := fmt.Sprintf("TextureSampler_%d", *samplerIndex)
			*samplerIndex++
			handle := pb.emitSampler(te.view.Proxy().TextureType(), te.view.Swizzle(), name)
			if !handle.IsValid() {
				ok = false
				return
			}
			vimpl.(*textureEffectImpl).setSamplerHandle(handle)
		}
	})
	return ok
}

// invokeFP builds the call expression that invokes fp's emitted function with its input color (and destination color,
// for blend functions) and coordinates.
func (pb *ProgramBuilder) invokeFP(fp FragmentProcessor, impl FPProgramImpl, inputColor, destColor, coords string) string {
	if fp.fpBase().IsBlendFunction() {
		if pb.fragmentProcessorHasCoordsParam(fp) {
			return fmt.Sprintf("%s(%s, %s, %s)", impl.fpImplBase().FunctionName(), inputColor,
				destColor, coords)
		}
		return fmt.Sprintf("%s(%s, %s)", impl.fpImplBase().FunctionName(), inputColor, destColor)
	}
	if pb.fragmentProcessorHasCoordsParam(fp) {
		return fmt.Sprintf("%s(%s, %s)", impl.fpImplBase().FunctionName(), inputColor, coords)
	}
	return fmt.Sprintf("%s(%s)", impl.fpImplBase().FunctionName(), inputColor)
}

// writeChildFPFunctions emits each of fp's child fragment processors as its own function, tracking substage indices so
// nested child names stay unique.
func (pb *ProgramBuilder) writeChildFPFunctions(fp FragmentProcessor, impl FPProgramImpl) {
	pb.substageIndices = append(pb.substageIndices, 0)
	for i := 0; i < impl.fpImplBase().NumChildProcessors(); i++ {
		childImpl := impl.fpImplBase().ChildProcessor(i)
		if childImpl == nil {
			continue
		}
		childFP := fp.fpBase().ChildProcessor(i)
		if childFP == nil {
			panic("impl child without FP child")
		}
		pb.writeFPFunction(childFP, childImpl)
		pb.substageIndices[len(pb.substageIndices)-1]++
	}
	pb.substageIndices = pb.substageIndices[:len(pb.substageIndices)-1]
}

// writeFPFunction emits fp's function (children first, so their uniforms are all registered).
func (pb *ProgramBuilder) writeFPFunction(fp FragmentProcessor, impl FPProgramImpl) {
	const dstColorParam = "_dst"
	inputColorParam := "_input"
	if fp.fpBase().IsBlendFunction() {
		inputColorParam = "_src"
	}
	sampleCoords := "_coords"
	pb.fs.nextStage()

	// An FP is always sampled at a particular coordinate, but if it is only sampled by a chain of uniform matrix
	// expressions the value that would have been passed to _coords is lifted to the vertex shader and a varying; then
	// the function does not take a _coords parameter.
	params := make([]ShaderVar, 0, 3)
	params = append(params, NewShaderVar(inputColorParam, GLSLTypeHalf4))
	if fp.fpBase().IsBlendFunction() {
		// Blend functions take a dest color as input.
		params = append(params, NewShaderVar(dstColorParam, GLSLTypeHalf4))
	}

	coordsEntry, inMap := pb.fpCoordsMap[fp]
	switch {
	case !inMap:
		// This FP isn't in the coords map at all, so its coords (if any) couldn't have been lifted to a varying.
		if fp.fpBase().UsesSampleCoords() {
			params = append(params, NewShaderVar(sampleCoords, GLSLTypeFloat2))
		}
	case coordsEntry.HasCoordsParam:
		// This FP is in the map, and it takes an explicit coords param.
		params = append(params, NewShaderVar(sampleCoords, GLSLTypeFloat2))
	default:
		// Either doesn't use coords at all or sampled through a chain of passthrough/matrix usages. In the latter case
		// the coords come from a varying.
		varying := coordsEntry.CoordsVarying
		switch varying.Type() {
		case GLSLTypeVoid:
			if fp.fpBase().UsesSampleCoordsDirectly() {
				panic("directly used coords without a varying")
			}
		case GLSLTypeFloat2:
			// Just point the local coords to the varying.
			sampleCoords = varying.Name()
		case GLSLTypeFloat3:
			// Must perform the perspective divide in the frag shader based on the varying; add a local variable in
			// place of the absent coords parameter.
			pb.fs.CodeAppendf("vec2 %s = %s.xy / %s.z;\n", sampleCoords, varying.Name(),
				varying.Name())
		default:
			panic("unexpected varying type for coords")
		}
	}

	// First, emit every child's function: this must happen (even for children that aren't sampled) so that all of the
	// expected uniforms are registered.
	pb.writeChildFPFunctions(fp, impl)
	args := makeFPEmitArgs(pb.fs, pb.uniformHandler, pb.shaderCaps(), fp, inputColorParam,
		dstColorParam, sampleCoords)

	impl.EmitCode(args)
	impl.fpImplBase().SetFunctionName(pb.fs.GetMangledFunctionName(fp.Name()))

	pb.fs.EmitFunction(GLSLTypeHalf4, impl.fpImplBase().FunctionName(), params,
		pb.fs.stageCode())
	pb.fs.deleteStage()
}

// emitAndInstallXferProc builds and emits the transfer processor's shader code, wiring up its input color/coverage and
// output(s).
func (pb *ProgramBuilder) emitAndInstallXferProc(colorIn, coverageIn string) bool {
	// Program builders have a bit of state we need to clear with each effect.
	pb.advanceStage()

	if pb.xpImpl != nil {
		panic("XP impl already created")
	}
	xp := pb.pipeline().XferProcessor()
	pb.xpImpl = xp.MakeProgramImpl()

	// Enable dual source secondary output if we have one.
	if xpHasSecondaryOutput(xp) {
		pb.fs.EnableSecondaryOutput()
	}

	pb.fs.CodeAppend("{\n")

	finalInColor := colorIn
	if finalInColor == "" {
		finalInColor = glslOpaqueWhite
	}
	finalInCoverage := coverageIn
	if finalInCoverage == "" {
		finalInCoverage = glslOpaqueWhite
	}

	args := &XPEmitArgs{
		FragBuilder:             pb.fs,
		UniformHandler:          pb.uniformHandler,
		ShaderCaps:              pb.shaderCaps(),
		XP:                      xp,
		InputColor:              finalInColor,
		InputCoverage:           finalInCoverage,
		OutputPrimary:           pb.fs.PrimaryColorOutputName(),
		OutputSecondary:         pb.fs.SecondaryColorOutputName(),
		DstTextureSamplerHandle: pb.dstTextureSamplerHandle,
		DstTextureOrigin:        pb.dstTextureOrigin,
		WriteSwizzle:            pb.pipeline().WriteSwizzle(),
	}
	xpImplEmitCode(pb.xpImpl, args)

	pb.fs.CodeAppend("}")
	return true
}

// emitSampler registers a new texture sampler uniform named name for the given texture type and read swizzle.
func (pb *ProgramBuilder) emitSampler(textureType gpu.TextureType, swizzle gpu.Swizzle, name string) SamplerHandle {
	pb.numFragmentSamplers++
	target := TEXTURE_2D
	if textureType == gpu.TextureTypeRectangle {
		target = TEXTURE_RECTANGLE
	}
	h, _ := pb.uniformHandler.AddSampler(uint32(target), swizzle, name)
	return h
}

// checkSamplerCounts reports whether the number of fragment samplers used is within the driver's limit.
func (pb *ProgramBuilder) checkSamplerCounts() bool {
	return pb.numFragmentSamplers <= pb.shaderCaps().MaxFragmentSamplers
}

// finalizeShaders finishes the varying declarations and concatenates the vertex/fragment shader sections into their
// final compiled GLSL sources.
func (pb *ProgramBuilder) finalizeShaders() {
	pb.varyingHandler.finish()
	pb.vs.finalize(ShaderFlagVertex)
	pb.fs.finalize(ShaderFlagFragment)
}

//////////////////////////////////////////////////////////////////////////////
// GL program creation: compiling, linking, and resolving locations.

// glStr returns a NUL-terminated copy of s for passing to GL entry points.
func glStr(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

// CreateProgram builds a complete GL program from a ProgramInfo: emits its shader source, compiles and links it, and
// resolves attribute/uniform locations. Returns nil on failure.
func CreateProgram(g *Gpu, desc *ProgramDesc, programInfo *ProgramInfo) *Program {
	builder := newProgramBuilder(g, desc, programInfo)
	if !builder.emitAndInstallProcs() {
		return nil
	}
	return builder.finalize()
}

// compileAndAttachShader compiles source as shaderType and attaches it to programID, panicking with the compiler's info
// log on failure; the new shader is appended to shadersToDelete for later cleanup. On a compile failure every GL object
// created so far — the failing shader, any already-attached shaders, and the program itself — is deleted before the
// panic, so a recovered panic leaks nothing (the same discipline the link-failure path in finalize follows).
func (pb *ProgramBuilder) compileAndAttachShader(programID, shaderType uint32, source []byte, shadersToDelete *[]uint32) bool {
	g := pb.gpu
	shaderID := g.fns().CreateShader(shaderType)
	if shaderID == 0 {
		return false
	}
	src := make([]byte, len(source)+1)
	copy(src, source)
	srcPtr := &src[0]
	length := int32(len(source))
	g.fns().ShaderSource(shaderID, 1, uintptr(unsafe.Pointer(&srcPtr)), &length)
	runtime.KeepAlive(src)

	g.programCache.stats.ShaderCompilations++
	g.fns().CompileShader(shaderID)

	var compiled int32
	g.fns().GetShaderiv(shaderID, COMPILE_STATUS, &compiled)
	if compiled == 0 {
		var infoLen int32
		g.fns().GetShaderiv(shaderID, INFO_LOG_LENGTH, &infoLen)
		log := ""
		if infoLen > 0 {
			buf := make([]byte, infoLen+1)
			var written int32
			g.fns().GetShaderInfoLog(shaderID, infoLen+1, &written, &buf[0])
			log = string(buf[:written])
		}
		g.fns().DeleteShader(shaderID)
		for _, id := range *shadersToDelete {
			g.fns().DeleteShader(id)
		}
		g.fns().DeleteProgram(programID)
		panic("shader compile failed: " + log + "\n" + string(source))
	}

	g.fns().AttachShader(programID, shaderID)
	*shadersToDelete = append(*shadersToDelete, shaderID)
	return true
}

// computeCountsAndStrides resolves the geometry processor's vertex/instance attributes into their final GL attribute
// layout, optionally binding each attribute's location explicitly.
func (pb *ProgramBuilder) computeCountsAndStrides(programID uint32, geomProc GeometryProcessor, bindAttribLocations bool) {
	base := geomProc.gpBase()
	pb.vertexAttributeCnt = base.NumVertexAttributes()
	pb.instanceAttributeCnt = base.NumInstanceAttributes()
	pb.attributes = make([]ProgramAttribute, 0, pb.vertexAttributeCnt+pb.instanceAttributeCnt)
	addAttr := func(i int, a *Attribute) {
		pb.attributes = append(pb.attributes, ProgramAttribute{
			CPUType:  a.CPUType(),
			GPUType:  a.GPUType(),
			Offset:   uintptr(a.offset),
			Location: i,
		})
		if bindAttribLocations {
			pb.gpu.fns().BindAttribLocation(programID, uint32(i), glStr(a.Name()))
		}
	}
	pb.vertexStride = base.VertexStride()
	i := 0
	for _, attr := range base.vertexAttributes.All() {
		a := attr
		addAttr(i, &a)
		i++
	}
	pb.instanceStride = base.InstanceStride()
	for _, attr := range base.instanceAttributes.All() {
		a := attr
		addAttr(i, &a)
		i++
	}
}

// bindProgramResourceLocations resolves uniform locations and, where the driver requires it, explicitly binds the
// fragment color output location(s) before linking.
func (pb *ProgramBuilder) bindProgramResourceLocations(programID uint32) {
	pb.uniformHandler.bindUniformLocations(programID, pb.gpu.glCaps())

	caps := pb.gpu.glCaps()
	if caps.BindFragDataLocationSupport() {
		if !caps.ShaderCaps.MustDeclareFragmentShaderOutput() {
			panic("bindFragDataLocation requires declared outputs")
		}
		pb.gpu.fns().BindFragDataLocation(programID, 0, glStr(declaredColorOutputName))
		if pb.fs.HasSecondaryOutput() {
			pb.gpu.fns().BindFragDataLocationIndexed(programID, 0, 1,
				glStr(declaredSecondaryColorOutputName))
		}
	}
}

// finalize compiles and links the shaders, resolves attribute and uniform locations, and wraps the result in a
// Program.
func (pb *ProgramBuilder) finalize() *Program {
	g := pb.gpu
	programID := g.fns().CreateProgram()
	if programID == 0 {
		return nil
	}

	pb.finalizeShaders()

	geomProc := pb.geometryProcessor()
	var shadersToDelete []uint32

	if !pb.compileAndAttachShader(programID, FRAGMENT_SHADER, pb.fs.compiled,
		&shadersToDelete) {
		g.fns().DeleteProgram(programID)
		return nil
	}
	if !pb.compileAndAttachShader(programID, VERTEX_SHADER, pb.vs.compiled, &shadersToDelete) {
		for _, id := range shadersToDelete {
			g.fns().DeleteShader(id)
		}
		g.fns().DeleteProgram(programID)
		return nil
	}

	// This also binds vertex attribute locations.
	pb.computeCountsAndStrides(programID, geomProc, true)
	pb.bindProgramResourceLocations(programID)

	g.fns().LinkProgram(programID)
	var linked int32
	g.fns().GetProgramiv(programID, LINK_STATUS, &linked)
	if linked == 0 {
		var infoLen int32
		g.fns().GetProgramiv(programID, INFO_LOG_LENGTH, &infoLen)
		log := ""
		if infoLen > 0 {
			buf := make([]byte, infoLen+1)
			var written int32
			g.fns().GetProgramInfoLog(programID, infoLen+1, &written, &buf[0])
			log = string(buf[:written])
		}
		for _, id := range shadersToDelete {
			g.fns().DeleteShader(id)
		}
		g.fns().DeleteProgram(programID)
		panic("program link failed: " + log + "\n" + string(pb.vs.compiled) + "\n" +
			string(pb.fs.compiled))
	}

	pb.uniformHandler.getUniformLocations(programID, g)

	for _, id := range shadersToDelete {
		g.fns().DeleteShader(id)
	}

	return pb.createProgram(programID)
}

// createProgram wraps a successfully linked program ID and this builder's accumulated state into a Program.
func (pb *ProgramBuilder) createProgram(programID uint32) *Program {
	return newGLProgram(pb.gpu, pb.uniformHandles, programID, pb.uniformHandler.uniforms,
		pb.uniformHandler.samplers, pb.gpImpl, pb.xpImpl, pb.fpImpls, pb.attributes,
		pb.vertexAttributeCnt, pb.instanceAttributeCnt, pb.vertexStride, pb.instanceStride,
		string(pb.vs.compiled), string(pb.fs.compiled))
}
