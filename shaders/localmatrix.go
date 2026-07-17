// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// LocalMatrixShader wraps a shader with a local matrix.

package shaders

import "github.com/richardwilkes/canvas/geom"

// LocalMatrixShader applies an extra local matrix in front of a wrapped shader.
type LocalMatrixShader struct {
	wrapped     Shader
	localMatrix geom.Matrix
}

// NewWithLocalMatrix wraps s with localMatrix: identity returns the shader unchanged, and wrapping an already-wrapped
// LocalMatrixShader concatenates into a single wrapper (lm' = localMatrix * innerLM).
func NewWithLocalMatrix(s Shader, localMatrix geom.Matrix) Shader {
	if s == nil {
		return nil
	}
	if localMatrix.IsIdentity() {
		return s
	}
	lm := localMatrix
	if inner, ok := s.(*LocalMatrixShader); ok {
		var m geom.Matrix
		m.SetConcat(&lm, &inner.localMatrix)
		lm = m
		s = inner.wrapped
	}
	return &LocalMatrixShader{wrapped: s, localMatrix: lm}
}

// Wrapped returns the wrapped shader; LocalMatrix returns the wrapper's matrix — the descriptor the GPU side consumes.
func (s *LocalMatrixShader) Wrapped() Shader { return s.wrapped }

// LocalMatrix returns the local matrix.
func (s *LocalMatrixShader) LocalMatrix() geom.Matrix { return s.localMatrix }

// IsOpaque implements Shader by delegating to the wrapped shader.
func (s *LocalMatrixShader) IsOpaque() bool { return s.wrapped.IsOpaque() }

// IsConstant implements Shader by delegating to the wrapped shader.
func (s *LocalMatrixShader) IsConstant() bool { return s.wrapped.IsConstant() }

// appendStages concatenates the local matrix into m and appends the wrapped shader's stages.
func (s *LocalMatrixShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	return s.wrapped.appendStages(p, m.concat(&s.localMatrix))
}
