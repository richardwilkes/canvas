// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build !darwin && !linux && !windows

package gltest

import "errors"

type platformContext struct{}

func (p *platformContext) init() error {
	// Only the darwin (CGL), Linux (GLX), and Windows (WGL) legs exist; GPU tests skip on every other platform.
	return errors.New("gltest: headless GL context creation is not implemented on this platform yet")
}

func (p *platformContext) destroy() {}
