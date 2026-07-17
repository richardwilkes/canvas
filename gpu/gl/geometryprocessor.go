// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GeometryProcessor implementations describe the geometric primitive being drawn: vertex/instance attribute layouts,
// texture samplers, and a ProgramImpl that emits the vertex shader, lifts fragment-processor coordinate transforms into
// varyings (collectTransforms/emitTransformCode), and writes the device position.

package gl

import (
	"fmt"
	"sort"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// AttributeAlignOffset rounds offset up to the next 4-byte boundary.
func AttributeAlignOffset(offset int) int { return (offset + 3) &^ 3 }

// attributeImplicitOffset is the sentinel offset meaning "assign this attribute's offset from ordering, not
// explicitly." 1 is safe to use since it is never a valid 4-byte-aligned offset.
const attributeImplicitOffset = 1

// Attribute describes a single vertex or instance attribute: its shader variable name, CPU-side storage type, GPU-side
// shader type, and byte offset within the vertex/instance stride.
type Attribute struct {
	name    string
	cpuType gpu.VertexAttribType
	gpuType GLSLType
	offset  int
}

// MakeAttribute makes an attribute whose offset is implicitly determined by the types and ordering of an attribute
// array.
func MakeAttribute(name string, cpuType gpu.VertexAttribType, gpuType GLSLType) Attribute {
	if name == "" || gpuType == GLSLTypeVoid {
		panic("attribute requires name and type")
	}
	return Attribute{
		name: name, cpuType: cpuType, gpuType: gpuType,
		offset: attributeImplicitOffset,
	}
}

// MakeAttributeWithOffset makes an attribute with an explicit offset.
func MakeAttributeWithOffset(name string, cpuType gpu.VertexAttribType, gpuType GLSLType, offset int) Attribute {
	if name == "" || gpuType == GLSLTypeVoid {
		panic("attribute requires name and type")
	}
	if AttributeAlignOffset(offset) != offset {
		panic("attribute offset must be 4-byte aligned")
	}
	return Attribute{name: name, cpuType: cpuType, gpuType: gpuType, offset: offset}
}

// IsInitialized reports whether the attribute has been given a name and type.
func (a *Attribute) IsInitialized() bool { return a.gpuType != GLSLTypeVoid }

// Name returns the attribute's shader variable name.
func (a *Attribute) Name() string { return a.name }

// CPUType returns the attribute's CPU-side storage type.
func (a *Attribute) CPUType() gpu.VertexAttribType { return a.cpuType }

// GPUType returns the attribute's GPU-side shader type.
func (a *Attribute) GPUType() GLSLType { return a.gpuType }

// Size returns the attribute's size in bytes.
func (a *Attribute) Size() int { return gpu.VertexAttribTypeSize(a.cpuType) }

// AsShaderVar returns the attribute as an "in" shader variable declaration.
func (a *Attribute) AsShaderVar() ShaderVar {
	return NewShaderVarMod(a.name, a.gpuType, TypeModifierIn)
}

// MakeColorAttribute builds a color attribute: 4 normalized unsigned bytes, or 4 floats when wideColor is set for
// higher precision.
func MakeColorAttribute(name string, wideColor bool) Attribute {
	cpuType := gpu.VertexAttribTypeUByte4Norm
	if wideColor {
		cpuType = gpu.VertexAttribTypeFloat4
	}
	return MakeAttribute(name, cpuType, GLSLTypeHalf4)
}

// AttributeSet is an ordered collection of vertex or instance attributes sharing one stride.
type AttributeSet struct {
	attributes []Attribute
	rawCount   int
	count      int
	stride     int
}

// InitImplicit assigns offsets from attribute ordering and computes the stride; uninitialized attributes are skipped by
// iteration.
func (s *AttributeSet) InitImplicit(attrs []Attribute) {
	s.attributes = attrs
	s.rawCount = len(attrs)
	s.count = 0
	s.stride = 0
	for i := range attrs {
		if attrs[i].IsInitialized() {
			s.count++
			s.stride += AttributeAlignOffset(attrs[i].Size())
		}
	}
}

// InitExplicit records an explicit stride and requires every attribute to already be initialized with its own aligned
// offset within it.
func (s *AttributeSet) InitExplicit(attrs []Attribute, stride int) {
	s.attributes = attrs
	s.rawCount = len(attrs)
	s.count = len(attrs)
	s.stride = stride
	if AttributeAlignOffset(stride) != stride {
		panic("stride must be 4-byte aligned")
	}
	for i := range attrs {
		if !attrs[i].IsInitialized() || attrs[i].offset == attributeImplicitOffset {
			panic("explicit attribute sets require initialized attributes with offsets")
		}
		if attrs[i].offset+attrs[i].Size() > stride {
			panic("attribute crosses stride boundary")
		}
	}
}

// Count returns the number of initialized attributes.
func (s *AttributeSet) Count() int { return s.count }

// Stride returns the byte stride between consecutive vertices/instances.
func (s *AttributeSet) Stride() int { return s.stride }

// All returns the initialized attributes with resolved offsets.
func (s *AttributeSet) All() []Attribute {
	out := make([]Attribute, 0, s.count)
	implicitOffset := 0
	for i := range s.attributes {
		a := s.attributes[i]
		if !a.IsInitialized() {
			continue
		}
		if a.offset == attributeImplicitOffset {
			a.offset = implicitOffset
			implicitOffset += AttributeAlignOffset(a.Size())
		}
		out = append(out, a)
	}
	return out
}

// AddToKey appends the attribute set's layout (stride, count, and per-attribute type/offset) to the shader program
// cache key.
func (s *AttributeSet) AddToKey(b *gpu.KeyBuilder) {
	b.AddBits(16, uint32(s.stride), "stride")
	b.AddBits(16, uint32(s.rawCount), "attribute count")
	implicitOffset := 0
	for i := range s.attributes {
		attr := &s.attributes[i]
		if attr.IsInitialized() {
			b.AppendComment(attr.Name())
			b.AddBits(8, uint32(attr.cpuType), "attrType")
			b.AddBits(8, uint32(attr.gpuType), "attrGpuType")
			offset := attr.offset
			if offset == attributeImplicitOffset {
				offset = implicitOffset
				implicitOffset += AttributeAlignOffset(attr.Size())
			}
			b.AddBits(16, uint32(uint16(offset)), "attrOffset")
		} else {
			b.AppendComment("unusedAttr")
			b.AddBits(8, 0xff, "attrType")
			b.AddBits(8, 0xff, "attrGpuType")
			b.AddBits(16, uint32(uint16(0xffff)), "attrOffset")
		}
	}
}

// TextureSampler describes one texture sampler used by a geometry processor; the actual texture proxies are stored in
// the draw op's fixed/dynamic state and passed to BindTextures at execute time.
type TextureSampler struct {
	samplerState gpu.SamplerState
	textureType  gpu.TextureType
	swizzle      gpu.Swizzle
	initialized  bool
}

// MakeGPTextureSampler builds a texture sampler description from a sampler state, texture type, and color swizzle.
func MakeGPTextureSampler(state gpu.SamplerState, textureType gpu.TextureType, swizzle gpu.Swizzle) TextureSampler {
	return TextureSampler{
		samplerState: state,
		textureType:  textureType,
		swizzle:      swizzle,
		initialized:  true,
	}
}

// SamplerState returns the sampler's filtering/wrap configuration.
func (t *TextureSampler) SamplerState() gpu.SamplerState { return t.samplerState }

// TextureType returns the sampler's expected texture type.
func (t *TextureSampler) TextureType() gpu.TextureType { return t.textureType }

// Swizzle returns the sampler's color-channel swizzle.
func (t *TextureSampler) Swizzle() gpu.Swizzle { return t.swizzle }

// IsInitialized reports whether the sampler has been configured.
func (t *TextureSampler) IsInitialized() bool { return t.initialized }

//////////////////////////////////////////////////////////////////////////////

// GeometryProcessor describes the geometric primitive being drawn: its vertex/instance attribute layout and how it
// emits vertex-shader code.
type GeometryProcessor interface {
	Processor

	// AddToKey appends this processor's identity and configuration to the shader program cache key.
	AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder)
	// MakeProgramImpl builds the ProgramImpl that emits this processor's shader code.
	MakeProgramImpl(caps *gpu.ShaderCaps) GPProgramImpl

	// gpBase gives shared code access to the embedded base.
	gpBase() *GPBase
}

// GPBase holds the fields shared by all GeometryProcessor implementations: vertex/instance attribute sets and texture
// samplers.
type GPBase struct {
	textureSamplers    []TextureSampler
	vertexAttributes   AttributeSet
	instanceAttributes AttributeSet
	processorBase
}

// initGP initializes the base with the processor's class ID.
func (b *GPBase) initGP(classID ClassID) { b.classID = classID }

func (b *GPBase) gpBase() *GPBase { return b }

// NumTextureSamplers returns the number of texture samplers this processor uses.
func (b *GPBase) NumTextureSamplers() int { return len(b.textureSamplers) }

// TextureSampler returns the i'th texture sampler.
func (b *GPBase) TextureSampler(i int) *TextureSampler { return &b.textureSamplers[i] }

// setTextureSamplers records the processor's texture samplers.
func (b *GPBase) setTextureSamplers(samplers []TextureSampler) { b.textureSamplers = samplers }

// NumVertexAttributes returns the number of initialized vertex attributes.
func (b *GPBase) NumVertexAttributes() int { return b.vertexAttributes.Count() }

// VertexAttributes returns the processor's vertex attribute set.
func (b *GPBase) VertexAttributes() *AttributeSet { return &b.vertexAttributes }

// NumInstanceAttributes returns the number of initialized instance attributes.
func (b *GPBase) NumInstanceAttributes() int { return b.instanceAttributes.Count() }

// InstanceAttributes returns the processor's instance attribute set.
func (b *GPBase) InstanceAttributes() *AttributeSet { return &b.instanceAttributes }

// HasVertexAttributes reports whether the processor has any vertex attributes.
func (b *GPBase) HasVertexAttributes() bool { return b.vertexAttributes.Count() > 0 }

// HasInstanceAttributes reports whether the processor has any instance attributes.
func (b *GPBase) HasInstanceAttributes() bool { return b.instanceAttributes.Count() > 0 }

// VertexStride returns the byte stride between vertices.
func (b *GPBase) VertexStride() int { return b.vertexAttributes.Stride() }

// InstanceStride returns the byte stride between instances.
func (b *GPBase) InstanceStride() int { return b.instanceAttributes.Stride() }

// setVertexAttributesWithImplicitOffsets sets the vertex attributes, assigning offsets from ordering.
func (b *GPBase) setVertexAttributesWithImplicitOffsets(attrs []Attribute) {
	b.vertexAttributes.InitImplicit(attrs)
}

// setInstanceAttributesWithImplicitOffsets sets the instance attributes, assigning offsets from ordering.
func (b *GPBase) setInstanceAttributesWithImplicitOffsets(attrs []Attribute) {
	b.instanceAttributes.InitImplicit(attrs)
}

// gpAttributeKey appends a geometry processor's vertex and instance attribute layouts to the shader program cache key.
func gpAttributeKey(gp GeometryProcessor, b *gpu.KeyBuilder) {
	b.AppendComment("vertex attributes")
	gp.gpBase().vertexAttributes.AddToKey(b)
	b.AppendComment("instance attributes")
	gp.gpBase().instanceAttributes.AddToKey(b)
}

// coordTransformKeyBits is the number of key bits used to encode one fragment processor's sample-coordinate transform
// kind.
const coordTransformKeyBits = 4

// ComputeCoordTransformsKey encodes a fragment processor's sample-coordinate usage kind (and whether it has
// perspective) for the shader program cache key. This is highly coupled with collectTransforms.
func ComputeCoordTransformsKey(fp FragmentProcessor) uint32 {
	key := uint32(fp.fpBase().SampleUsage().Kind()) << 1
	if fp.fpBase().SampleUsage().HasPerspective() {
		key |= 0b1
	}
	return key
}

//////////////////////////////////////////////////////////////////////////////

// FPCoords records how a fragment processor obtains its sample coordinates: through a varying, a coords parameter, or
// both.
type FPCoords struct {
	CoordsVarying  ShaderVar
	HasCoordsParam bool
}

// FPCoordsMap maps each fragment processor in a pipeline to its resolved sample-coordinate source.
type FPCoordsMap map[FragmentProcessor]FPCoords

// GPEmitArgs bundles the inputs a GeometryProcessor's ProgramImpl needs to emit its shader code.
type GPEmitArgs struct {
	VertBuilder    *VertexShaderBuilder
	FragBuilder    *FragmentShaderBuilder
	VaryingHandler *VaryingHandler
	UniformHandler *UniformHandler
	ShaderCaps     *gpu.ShaderCaps
	GeomProc       GeometryProcessor
	OutputColor    string
	OutputCoverage string
	TexSamplers    []SamplerHandle
}

// GPArgs holds the outputs of onEmitCode.
type GPArgs struct {
	// PositionVar is the variable the GP stores its device position in (float2 or float3).
	PositionVar ShaderVar
	// LocalCoordVar is the variable storing the draw's local coordinates (float2, float3, or void when no FP needs
	// local coordinates). It can be an attribute or local variable but not itself a varying.
	LocalCoordVar ShaderVar
	// LocalCoordShaderIsFragment is true when the local coordinates are computed in the fragment shader rather than the
	// vertex shader (the vertex shader is preferable when either is possible).
	LocalCoordShaderIsFragment bool
}

// GPProgramImpl emits the shader code for a GeometryProcessor and uploads its uniform values.
type GPProgramImpl interface {
	// SetData uploads the processor's per-draw uniform values.
	SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor)
	// onEmitCode emits the vertex/fragment shader code for this processor.
	onEmitCode(args *GPEmitArgs, gpArgs *GPArgs)
	// gpImplBase gives shared code access to the embedded base.
	gpImplBase() *GPImplBase
}

// gpTransformInfo records a lifted coordinate-transform varying and the fragment-processor traversal order it was
// created at.
type gpTransformInfo struct {
	varying        Varying
	inputCoords    ShaderVar
	traversalOrder int
}

// GPImplBase holds the fields shared by all GPProgramImpl implementations: the map of lifted coordinate-transform
// varyings, keyed by the fragment processor whose sample matrix produced them.
type GPImplBase struct {
	transformVaryingsMap map[FragmentProcessor]*gpTransformInfo
}

func (b *GPImplBase) gpImplBase() *GPImplBase { return b }

// gpImplEmitCode runs onEmitCode, collects fragment-processor coordinate transforms, and writes the normalized device
// position.
func gpImplEmitCode(impl GPProgramImpl, args *GPEmitArgs, pipeline *Pipeline) (FPCoordsMap, ShaderVar) {
	var gpArgs GPArgs
	impl.onEmitCode(args, &gpArgs)

	transformMap := impl.gpImplBase().collectTransforms(args.VertBuilder, args.VaryingHandler, &gpArgs, pipeline)

	if gpArgs.PositionVar.Type() != GLSLTypeFloat2 && gpArgs.PositionVar.Type() != GLSLTypeFloat3 {
		panic("position must be float2 or float3")
	}
	args.VertBuilder.EmitNormalizedPosition(gpArgs.PositionVar.Name(), gpArgs.PositionVar.Type())
	if gpArgs.PositionVar.Type() == GLSLTypeFloat2 {
		args.VaryingHandler.SetNoPerspective()
	}

	return transformMap, gpArgs.LocalCoordVar
}

// baseCoord identifies which base coordinate system a fragment processor's sample matrix chain is anchored to.
type baseCoord int8

const (
	baseCoordNone baseCoord = iota
	baseCoordLocal
	baseCoordPosition
)

// collectTransforms performs a pre-order traversal of the fragment-processor trees that identifies FPs sampled with a
// series of matrices applied to local coords and lifts those coords into varyings.
func (b *GPImplBase) collectTransforms(vb *VertexShaderBuilder, varyingHandler *VaryingHandler, gpArgs *GPArgs, pipeline *Pipeline) FPCoordsMap {
	localCoordsVar := gpArgs.LocalCoordVar
	positionVar := gpArgs.PositionVar
	localCoordsInFragment := gpArgs.LocalCoordShaderIsFragment

	switch localCoordsVar.Type() {
	case GLSLTypeFloat2, GLSLTypeFloat3, GLSLTypeVoid:
	default:
		panic("local coords must be float2/float3/void")
	}
	switch positionVar.Type() {
	case GLSLTypeFloat2, GLSLTypeFloat3, GLSLTypeVoid:
	default:
		panic("position must be float2/float3/void")
	}

	if b.transformVaryingsMap == nil {
		b.transformVaryingsMap = make(map[FragmentProcessor]*gpTransformInfo)
	}

	var baseLocalCoordVarying Varying
	baseLocalCoordFSVar := func() ShaderVar {
		if localCoordsInFragment {
			return localCoordsVar
		}
		if !localCoordsVar.Type().IsFloatType() {
			panic("vertex-shader local coords must be a float type")
		}
		if baseLocalCoordVarying.Type() == GLSLTypeVoid {
			// Initialize to the GP-provided coordinate.
			baseLocalCoordVarying = NewVarying(localCoordsVar.Type())
			varyingHandler.AddVarying("LocalCoord", &baseLocalCoordVarying)
			vb.CodeAppendf("%s = %s;\n", baseLocalCoordVarying.VsOut(), localCoordsVar.Name())
		}
		return baseLocalCoordVarying.FsInVar()
	}

	canUsePosition := positionVar.Type() != GLSLTypeVoid

	result := make(FPCoordsMap)
	traversalIndex := 0
	var liftTransforms func(fp FragmentProcessor, hasPerspective bool,
		lastMatrixFP FragmentProcessor, lastMatrixTraversalIndex int, bc baseCoord)
	liftTransforms = func(fp FragmentProcessor, hasPerspective bool,
		lastMatrixFP FragmentProcessor, lastMatrixTraversalIndex int, bc baseCoord,
	) {
		traversalIndex++
		if !localCoordsInFragment {
			switch fp.fpBase().SampleUsage().Kind() {
			case SampleUsageNone:
				// This should only happen at the root.
				if fp.fpBase().Parent() != nil {
					panic("unsampled non-root FP")
				}
			case SampleUsagePassThrough:
			case SampleUsageUniformMatrix:
				// Update tracking of the last matrix and matrix props.
				hasPerspective = hasPerspective || fp.fpBase().SampleUsage().HasPerspective()
				lastMatrixFP = fp
				lastMatrixTraversalIndex = traversalIndex
			case SampleUsageFragCoord:
				hasPerspective = positionVar.Type() == GLSLTypeFloat3
				lastMatrixFP = nil
				lastMatrixTraversalIndex = -1
				bc = baseCoordPosition
			case SampleUsageExplicit:
				bc = baseCoordNone
			}
		} else {
			// If the GP doesn't provide an interpolatable local coord then there is no hope to lift.
			bc = baseCoordNone
		}

		entry := FPCoords{HasCoordsParam: fp.fpBase().UsesSampleCoordsDirectly()}

		// We add a varying if we're in a chain of matrices multiplied by local or device coords. If the coord is the
		// untransformed local coord we add a varying. We don't for untransformed device coords since that saves nothing
		// over "gl_FragCoord.xy". And if the FP doesn't directly use its coords we don't add a varying.
		if fp.fpBase().UsesSampleCoordsDirectly() &&
			(bc == baseCoordLocal ||
				(bc == baseCoordPosition && lastMatrixFP != nil && canUsePosition)) {
			// Associate the varying with the highest possible node in the FP tree that shares the same coordinates so
			// multiple FPs in a subtree can share.
			if lastMatrixFP == nil {
				entry.CoordsVarying = baseLocalCoordFSVar()
			} else {
				info := b.transformVaryingsMap[lastMatrixFP]
				if info == nil {
					t := GLSLTypeFloat2
					if hasPerspective {
						t = GLSLTypeFloat3
					}
					info = &gpTransformInfo{varying: NewVarying(t)}
					varyingHandler.AddVarying(
						fmt.Sprintf("TransformedCoords_%d", lastMatrixTraversalIndex),
						&info.varying,
					)
					if bc == baseCoordLocal {
						info.inputCoords = localCoordsVar
					} else {
						info.inputCoords = positionVar
					}
					info.traversalOrder = lastMatrixTraversalIndex
					b.transformVaryingsMap[lastMatrixFP] = info
				}
				if info.traversalOrder != lastMatrixTraversalIndex {
					panic("varying traversal order mismatch")
				}
				// The FP will use the varying in the fragment shader as its coords.
				entry.CoordsVarying = info.varying.FsInVar()
			}
			entry.HasCoordsParam = false
		}
		result[fp] = entry

		for c := 0; c < fp.fpBase().NumChildProcessors(); c++ {
			if child := fp.fpBase().ChildProcessor(c); child != nil {
				liftTransforms(child, hasPerspective, lastMatrixFP, lastMatrixTraversalIndex, bc)
				// If we have a varying then we never need a param. Otherwise, if one of our children takes a
				// non-explicit coord then we'll need our coord.
				entry = result[fp]
				if entry.CoordsVarying.Type() == GLSLTypeVoid &&
					!child.fpBase().SampleUsage().IsExplicit() &&
					!child.fpBase().SampleUsage().IsFragCoord() &&
					result[child].HasCoordsParam {
					entry.HasCoordsParam = true
					result[fp] = entry
				}
			}
		}
	}

	hasPerspective := localCoordsVar.Type().VecLength() == 3
	for i := 0; i < pipeline.NumFragmentProcessors(); i++ {
		liftTransforms(pipeline.FragmentProcessor(i), hasPerspective, nil, -1, baseCoordLocal)
	}
	return result
}

// emitTransformCode generates the vertex-shader assignments for the varyings recorded by collectTransforms. Runs after
// fragment-processor code emission so the uniforms registered for matrix sampling already exist.
func (b *GPImplBase) emitTransformCode(vb *VertexShaderBuilder, uniformHandler *UniformHandler) {
	// Visit the varyings in FP pre-order traversal order so descendant varyings can be computed from ancestor varyings.
	type fpAndInfo struct {
		fp   FragmentProcessor
		info *gpTransformInfo
	}
	entries := make([]fpAndInfo, 0, len(b.transformVaryingsMap))
	for fp, info := range b.transformVaryingsMap {
		entries = append(entries, fpAndInfo{fp: fp, info: info})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].info.traversalOrder < entries[j].info.traversalOrder
	})
	for _, e := range entries {
		fp, info := e.fp, e.info
		if !fp.fpBase().SampleUsage().IsUniformMatrix() {
			panic("recorded transform info requires a uniform sample matrix")
		}
		uniform := uniformHandler.liftUniformToVertexShader(fp.fpBase().Parent(),
			MatrixUniformName())
		// Start with this matrix and accumulate additional matrices walking up the FP tree to either the base coords or
		// an ancestor FP with an associated varying.
		transformExpression := uniform.Name()
		inputCoords := info.inputCoords

		for base := fp.fpBase().Parent(); base != nil; base = base.fpBase().Parent() {
			if baseInfo := b.transformVaryingsMap[base]; baseInfo != nil {
				// Can stop here: this varying already holds all transforms from higher FPs.
				inputCoords = baseInfo.varying.VsOutVar()
				break
			}
			switch {
			case base.fpBase().SampleUsage().IsUniformMatrix():
				parentUniform := uniformHandler.liftUniformToVertexShader(
					base.fpBase().Parent(), MatrixUniformName(),
				)
				transformExpression += " * " + parentUniform.Name()
			case base.fpBase().SampleUsage().IsFragCoord():
				// The chain of matrices starts here, based on the device-space position.
			default:
				// A pass-through FP that doesn't need to be built into the expression.
				if !base.fpBase().SampleUsage().IsPassThrough() &&
					base.fpBase().SampleUsage().IsSampled() {
					panic("unexpected sample usage in transform chain")
				}
				continue
			}
			if base.fpBase().SampleUsage().IsFragCoord() {
				break
			}
		}

		var inputStr string
		if inputCoords.Type() == GLSLTypeFloat2 {
			inputStr = fmt.Sprintf("vec3(%s, 1.0)", inputCoords.Name())
		} else {
			if inputCoords.Type() != GLSLTypeFloat3 {
				panic("unexpected input coord type")
			}
			inputStr = inputCoords.Name()
		}

		vb.CodeAppend("{\n")
		if info.varying.Type() == GLSLTypeFloat2 {
			if vb.programBuilder.gpu.Caps().ShaderCaps.NonsquareMatrixSupport {
				vb.CodeAppendf("%s = mat3x2(%s) * %s", info.varying.VsOut(),
					transformExpression, inputStr)
			} else {
				vb.CodeAppendf("%s = (%s * %s).xy", info.varying.VsOut(), transformExpression,
					inputStr)
			}
		} else {
			if info.varying.Type() != GLSLTypeFloat3 {
				panic("unexpected varying type")
			}
			vb.CodeAppendf("%s = %s * %s", info.varying.VsOut(), transformExpression, inputStr)
		}
		vb.CodeAppend(";\n")
		vb.CodeAppend("}\n")
	}
	// We don't need this map anymore.
	b.transformVaryingsMap = nil
}

// setupUniformColor emits a fragment-shader uniform color and assigns it to outputName.
func (b *GPImplBase) setupUniformColor(fragBuilder *FragmentShaderBuilder, uniformHandler *UniformHandler, outputName string, colorUniform *UniformHandle) {
	var stagedLocalVarName string
	*colorUniform, stagedLocalVarName = uniformHandler.AddUniform(nil, ShaderFlagFragment,
		GLSLTypeHalf4, "Color")
	fragBuilder.CodeAppendf("%s = %s;", outputName, stagedLocalVarName)
	if fragBuilder.programBuilder.gpu.Caps().ShaderCaps.MustObfuscateUniformColor {
		fragBuilder.CodeAppendf("%s = max(%s, vec4(0));", outputName, outputName)
	}
}

// matrixKeyBits is the number of key bits used to encode one matrix's ComputeMatrixKey classification.
const matrixKeyBits = 2

// ComputeMatrixKey classifies a matrix for the shader program cache key: identity, scale and translate only, affine, or
// perspective.
func ComputeMatrixKey(caps *gpu.ShaderCaps, mat *geom.Matrix) uint32 {
	if !caps.ReducedShaderMode {
		if mat.IsIdentity() {
			return 0b00
		}
		if mat.IsScaleTranslate() {
			return 0b01
		}
	}
	if !mat.HasPerspective() {
		return 0b10
	}
	return 0b11
}

// ComputeMatrixKeys packs the view and local matrix classifications into one key.
func ComputeMatrixKeys(caps *gpu.ShaderCaps, viewMatrix, localMatrix *geom.Matrix) uint32 {
	return ComputeMatrixKey(caps, viewMatrix)<<matrixKeyBits | ComputeMatrixKey(caps, localMatrix)
}

// AddMatrixKeys shifts flags left to make room for, and appends, the view and local matrix keys.
func AddMatrixKeys(caps *gpu.ShaderCaps, flags uint32, viewMatrix, localMatrix *geom.Matrix) uint32 {
	// Shifting to make room for the matrix keys must not lose bits.
	if flags<<(2*matrixKeyBits)>>(2*matrixKeyBits) != flags {
		panic("flags overflow matrix key shift")
	}
	return flags<<(2*matrixKeyBits) | ComputeMatrixKeys(caps, viewMatrix, localMatrix)
}

// SetTransform uploads a matrix uniform created by WriteOutputPosition/WriteLocalCoord, skipping unused uniforms and
// unchanged state.
func SetTransform(pdman *ProgramDataManager, caps *gpu.ShaderCaps, uniform UniformHandle, matrix, state *geom.Matrix, stateValid *bool) {
	if !uniform.IsValid() || (state != nil && *stateValid && *state == *matrix) {
		// No update needed.
		return
	}
	if state != nil {
		*state = *matrix
		*stateValid = true
	}
	if matrix.IsScaleTranslate() && !caps.ReducedShaderMode {
		// The uniform is a float4 in this mode (ComputeMatrixKey chose the compact form).
		pdman.Set4f(uniform, matrix.Get(0), matrix.Get(2), matrix.Get(4), matrix.Get(5))
	} else {
		pdman.SetMatrix(uniform, matrix)
	}
}

// writePassthroughVertexPosition copies an input position straight into a local variable, unmodified.
func writePassthroughVertexPosition(vertBuilder *VertexShaderBuilder, inPos ShaderVar, outPos *ShaderVar) {
	if inPos.Type() != GLSLTypeFloat2 && inPos.Type() != GLSLTypeFloat3 {
		panic("position must be float2 or float3")
	}
	outName := vertBuilder.NewTmpVarName(inPos.Name())
	outPos.Set(inPos.Type(), outName)
	vertBuilder.CodeAppendf("vec%d %s = %s;", inPos.Type().VecLength(), outName, inPos.Name())
}

// writeVertexPosition emits vertex-shader code that transforms inPos by matrix (uploaded as a uniform), choosing among
// passthrough, compact scale+translate, and full 3x3/perspective forms.
func writeVertexPosition(vertBuilder *VertexShaderBuilder, uniformHandler *UniformHandler, caps *gpu.ShaderCaps, inPos ShaderVar, matrix *geom.Matrix, matrixName string, outPos *ShaderVar, matrixUniform *UniformHandle) {
	if inPos.Type() != GLSLTypeFloat2 && inPos.Type() != GLSLTypeFloat3 {
		panic("position must be float2 or float3")
	}
	outName := vertBuilder.NewTmpVarName(inPos.Name())

	if matrix.IsIdentity() && !caps.ReducedShaderMode {
		writePassthroughVertexPosition(vertBuilder, inPos, outPos)
		return
	}

	useCompactTransform := matrix.IsScaleTranslate() && !caps.ReducedShaderMode
	uniType := GLSLTypeFloat3x3
	if useCompactTransform {
		uniType = GLSLTypeFloat4
	}
	var mangledMatrixName string
	*matrixUniform, mangledMatrixName = uniformHandler.AddUniform(nil, ShaderFlagVertex, uniType,
		matrixName)

	if inPos.Type() == GLSLTypeFloat3 {
		// A float3 stays a float3 whether or not the matrix adds perspective.
		if useCompactTransform {
			vertBuilder.CodeAppendf("vec3 %s = vec3(%s.xz, 1.0) * %s + vec3(%s.yw, 0.0);\n",
				outName, mangledMatrixName, inPos.Name(), mangledMatrixName)
		} else {
			vertBuilder.CodeAppendf("vec3 %s = %s * %s;\n", outName, mangledMatrixName,
				inPos.Name())
		}
		outPos.Set(GLSLTypeFloat3, outName)
		return
	}
	if matrix.HasPerspective() {
		// A float2 is promoted to a float3 if the matrix adds perspective.
		if useCompactTransform {
			panic("compact transform cannot have perspective")
		}
		vertBuilder.CodeAppendf("vec3 %s = (%s * vec3(%s, 1.0));", outName, mangledMatrixName,
			inPos.Name())
		outPos.Set(GLSLTypeFloat3, outName)
		return
	}
	switch {
	case useCompactTransform:
		vertBuilder.CodeAppendf("vec2 %s = %s.xz * %s + %s.yw;\n", outName, mangledMatrixName,
			inPos.Name(), mangledMatrixName)
	case caps.NonsquareMatrixSupport:
		vertBuilder.CodeAppendf("vec2 %s = mat3x2(%s) * vec3(%s, 1.0);\n", outName,
			mangledMatrixName, inPos.Name())
	default:
		vertBuilder.CodeAppendf("vec2 %s = (%s * vec3(%s, 1.0)).xy;\n", outName,
			mangledMatrixName, inPos.Name())
	}
	outPos.Set(GLSLTypeFloat2, outName)
}

// WriteOutputPosition writes the device position straight through from a float2 vertex attribute, with no matrix.
func WriteOutputPosition(vertBuilder *VertexShaderBuilder, gpArgs *GPArgs, posName string) {
	writePassthroughVertexPosition(vertBuilder, NewShaderVar(posName, GLSLTypeFloat2),
		&gpArgs.PositionVar)
}

// WriteOutputPositionWithMatrix writes the device position by transforming a float2 vertex attribute through
// viewMatrix.
func WriteOutputPositionWithMatrix(vertBuilder *VertexShaderBuilder, uniformHandler *UniformHandler, caps *gpu.ShaderCaps, gpArgs *GPArgs, posName string, viewMatrix *geom.Matrix, viewMatrixUniform *UniformHandle) {
	writeVertexPosition(vertBuilder, uniformHandler, caps,
		NewShaderVar(posName, GLSLTypeFloat2), viewMatrix, "viewMatrix", &gpArgs.PositionVar,
		viewMatrixUniform)
}

// WriteLocalCoord writes the local coordinate used by fragment processors, transforming localVar through localMatrix.
func WriteLocalCoord(vertBuilder *VertexShaderBuilder, uniformHandler *UniformHandler, caps *gpu.ShaderCaps, gpArgs *GPArgs, localVar ShaderVar, localMatrix *geom.Matrix, localMatrixUniform *UniformHandle) {
	writeVertexPosition(vertBuilder, uniformHandler, caps, localVar, localMatrix, "localMatrix",
		&gpArgs.LocalCoordVar, localMatrixUniform)
}
