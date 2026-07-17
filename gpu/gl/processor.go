// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ClassID enumerates every processor subclass (fragment processors, geometry processors, and transfer processors) in a
// fixed order; class IDs are folded into program keys with 8 bits, and keeping the ordering stable keeps our keys
// aligned with a reference numbering used by the GLSL parity parity harness.

package gl

// ClassID identifies a Processor's concrete implementation type.
type ClassID uint8

// ClassID values.
const (
	NullClassID ClassID = iota // reserved for missing (null) processors

	AttributeTestProcessorClassID
	BigKeyProcessorClassID
	BlendFragmentProcessorClassID
	BlockInputFragmentProcessorClassID
	ButtCapStrokedCircleGeometryProcessorClassID
	CircleGeometryProcessorClassID
	CircularRRectEffectClassID
	ClockwiseTestProcessorClassID
	ColorTableEffectClassID
	CoverageSetOpXPClassID
	CustomXPClassID
	DashingCircleEffectClassID
	DashingLineEffectClassID
	DefaultGeoProcClassID
	DeviceSpaceClassID
	DIEllipseGeometryProcessorClassID
	DisableColorXPClassID
	DrawAtlasPathShaderClassID
	EllipseGeometryProcessorClassID
	EllipticalRRectEffectClassID
	FwidthSquircleTestProcessorClassID
	GPClassID
	BicubicEffectClassID
	BitmapTextGeoProcClassID
	ColorSpaceXformEffectClassID
	ConicEffectClassID
	ConvexPolyEffectClassID
	DiffuseLightingEffectClassID
	DisplacementMapEffectClassID
	DistanceFieldA8TextGeoProcClassID
	DistanceFieldLCDTextGeoProcClassID
	DistanceFieldPathGeoProcClassID
	FillRRectOpProcessorClassID
	GaussianConvolutionFragmentProcessorClassID
	MatrixConvolutionEffectClassID
	MatrixEffectClassID
	MeshTestProcessorClassID
	MorphologyEffectClassID
	PerlinNoise2EffectClassID
	PipelineDynamicStateTestProcessorClassID
	QuadEffectClassID
	RRectShadowGeoProcClassID
	GLSLFPClassID
	SpecularLightingEffectClassID
	TextureEffectClassID
	UnrolledBinaryGradientColorizerClassID
	YUVtoRGBEffectClassID
	HighPrecisionFragmentProcessorClassID
	LatticeGPClassID
	PDLCDXferProcessorClassID
	PorterDuffXferProcessorClassID
	PremulFragmentProcessorClassID
	QuadEdgeEffectClassID
	QuadPerEdgeAAGeometryProcessorClassID
	SeriesFragmentProcessorClassID
	ShaderPDXferProcessorClassID
	SurfaceColorProcessorClassID
	SwizzleFragmentProcessorClassID
	TessellateBoundingBoxShaderClassID
	TessellateModulateAtlasCoverageEffectClassID
	TessellateStrokeTessellationShaderClassID
	TessellateHullShaderClassID
	TessellateMiddleOutShaderClassID
	TessellateSimpleTriangleShaderClassID
	TessellationTestTriShaderClassID
	TestFPClassID
	TestRectOpClassID
	VertexColorSpaceBenchGPClassID
	VerticesGPClassID
)

// Processor is the virtual surface shared by fragment processors, geometry processors, and transfer processors.
type Processor interface {
	// Name is a human-meaningful string to identify this processor; it may be embedded in generated shader code and
	// must be a legal GLSL identifier prefix.
	Name() string
	// ClassID identifies the processor subclass.
	ClassID() ClassID
}

// processorBase holds the state common to every Processor implementation.
type processorBase struct {
	classID ClassID
}

// ClassID implements Processor.
func (p *processorBase) ClassID() ClassID { return p.classID }

// SampleUsageKind classifies how a child fragment processor is invoked by its parent.
type SampleUsageKind int8

// SampleUsageKind values.
const (
	// SampleUsageNone: child is not sampled at all.
	SampleUsageNone SampleUsageKind = iota
	// SampleUsagePassThrough: child is sampled with the inherited coordinates.
	SampleUsagePassThrough
	// SampleUsageUniformMatrix: child is sampled with a matrix whose value is uniform.
	SampleUsageUniformMatrix
	// SampleUsageFragCoord: child is sampled using the fragment coordinate.
	SampleUsageFragCoord
	// SampleUsageExplicit: child is sampled at an explicit coordinate.
	SampleUsageExplicit
)

// SampleUsage describes how a fragment processor is invoked by its parent.
type SampleUsage struct {
	kind           SampleUsageKind
	hasPerspective bool // only valid for UniformMatrix
}

// PassThroughSampleUsage builds a SampleUsage for a child sampled with the inherited coordinates.
func PassThroughSampleUsage() SampleUsage { return SampleUsage{kind: SampleUsagePassThrough} }

// UniformMatrixSampleUsage builds a SampleUsage for a child sampled through a uniform matrix.
func UniformMatrixSampleUsage(hasPerspective bool) SampleUsage {
	return SampleUsage{kind: SampleUsageUniformMatrix, hasPerspective: hasPerspective}
}

// ExplicitSampleUsage builds a SampleUsage for a child sampled at an explicit coordinate.
func ExplicitSampleUsage() SampleUsage { return SampleUsage{kind: SampleUsageExplicit} }

// FragCoordSampleUsage builds a SampleUsage for a child sampled using the fragment coordinate.
func FragCoordSampleUsage() SampleUsage { return SampleUsage{kind: SampleUsageFragCoord} }

// MatrixUniformName is the (pre-mangling) name of the uniform every matrix-sampled child's transform is stored in.
func MatrixUniformName() string { return "matrix" }

// Kind returns the sample usage kind.
func (u SampleUsage) Kind() SampleUsageKind { return u.kind }

// HasPerspective reports whether the sampling matrix has perspective (only meaningful when Kind is
// SampleUsageUniformMatrix).
func (u SampleUsage) HasPerspective() bool { return u.hasPerspective }

// IsSampled reports whether the child is sampled at all.
func (u SampleUsage) IsSampled() bool { return u.kind != SampleUsageNone }

// IsPassThrough reports whether the child is sampled with the inherited coordinates.
func (u SampleUsage) IsPassThrough() bool { return u.kind == SampleUsagePassThrough }

// IsExplicit reports whether the child is sampled at an explicit coordinate.
func (u SampleUsage) IsExplicit() bool { return u.kind == SampleUsageExplicit }

// IsUniformMatrix reports whether the child is sampled through a uniform matrix.
func (u SampleUsage) IsUniformMatrix() bool { return u.kind == SampleUsageUniformMatrix }

// IsFragCoord reports whether the child is sampled using the fragment coordinate.
func (u SampleUsage) IsFragCoord() bool { return u.kind == SampleUsageFragCoord }
