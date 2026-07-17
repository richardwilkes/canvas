// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ColorShader (a constant-color shader) and the empty shader that renders nothing.

package shaders

import "github.com/richardwilkes/canvas/colorcore"

// ColorShader is a constant-color shader over a float color.
type ColorShader struct {
	color colorcore.Color4f
}

// NewColor builds a ColorShader from an unpremultiplied 8-bit color.
func NewColor(c colorcore.Color) *ColorShader {
	return &ColorShader{color: colorcore.Color4fFromColor(c)}
}

// NewColor4f builds a ColorShader from an unpremultiplied float color, which is interpreted as sRGB.
func NewColor4f(c colorcore.Color4f) *ColorShader {
	return &ColorShader{color: c}
}

// Color returns the shader's (unpremultiplied) color — the descriptor the GPU side consumes.
func (s *ColorShader) Color() colorcore.Color4f { return s.color }

// SetColor re-initializes s to the given color, the scratch-construction counterpart of NewColor: a caller reuses one
// ColorShader value across draws — the DrawAtlas per-sprite color lane — instead of allocating a fresh *ColorShader per
// sprite. Only for transient scratch shaders consumed synchronously within a draw.
func (s *ColorShader) SetColor(c colorcore.Color) { s.color = colorcore.Color4fFromColor(c) }

// IsOpaque implements Shader.
func (s *ColorShader) IsOpaque() bool { return s.color.IsOpaque() }

// IsConstant implements Shader: a solid color is always constant.
func (s *ColorShader) IsConstant() bool { return true }

// appendStages premultiplies the color (the sRGB→dst transform is the identity) and appends it as the pipeline's
// constant color.
func (s *ColorShader) appendStages(p *Pipeline, _ MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	p.AppendConstantColor(s.color.Premul())
	return true
}

///////////////////////////////////////////////////////////////////////////////

// emptyShader renders nothing (appendStages fails, so the blitter chooser produces a null blitter).
type emptyShader struct{}

// Empty returns the shader that renders nothing.
func Empty() Shader { return emptyShader{} }

// IsOpaque implements Shader.
func (emptyShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (emptyShader) IsConstant() bool { return false }

func (emptyShader) appendStages(_ *Pipeline, _ MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	return false
}

// IsEmpty reports whether s is the empty shader (the GPU fragment-processor conversion returns nil for it).
func IsEmpty(s Shader) bool {
	_, ok := s.(emptyShader)
	return ok
}
