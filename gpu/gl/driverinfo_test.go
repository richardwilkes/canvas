// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import "testing"

func TestGetVendor(t *testing.T) {
	cases := []struct {
		in   string
		want Vendor
	}{
		{in: "Apple", want: VendorApple},
		{in: "Intel", want: VendorIntel},
		{in: "Intel Inc.", want: VendorIntel},
		{in: "Intel Open Source Technology Center", want: VendorIntel},
		{in: "NVIDIA Corporation", want: VendorNVIDIA},
		{in: "ATI Technologies Inc.", want: VendorATI},
		{in: "ARM", want: VendorARM},
		{in: "Qualcomm", want: VendorQualcomm},
		{in: "freedreno", want: VendorQualcomm},
		{in: "Imagination Technologies", want: VendorImagination},
		{in: "Google Inc.", want: VendorGoogle},
		{in: "Mesa/X.org", want: VendorOther},
		{in: "AMD", want: VendorOther},
	}
	for _, c := range cases {
		if got := getVendor(c.in); got != c.want {
			t.Errorf("getVendor(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestGetRenderer(t *testing.T) {
	cases := []struct {
		in   string
		want Renderer
	}{
		{in: "Apple M4 Max", want: RendererApple},
		{in: "Apple M1", want: RendererApple},
		{in: "llvmpipe (LLVM 15.0.7, 256 bits)", want: RendererGalliumLLVM},
		{in: "Intel Iris OpenGL Engine", want: RendererIntelHaswell},
		{in: "Intel Iris Pro OpenGL Engine", want: RendererIntelHaswell},
		{in: "Intel(R) HD Graphics 630", want: RendererIntelKabyLake},
		{in: "Intel(R) UHD Graphics 630", want: RendererIntelCoffeeLake},
		{in: "Intel(R) Iris(TM) Plus Graphics 655", want: RendererIntelCoffeeLake},
		{in: "Intel(R) HD Graphics 4000", want: RendererIntelIvyBridge},
		{in: "Intel(R) Iris(R) Xe Graphics", want: RendererIntelTigerLake},
		{in: "Mesa Intel(R) UHD Graphics 770 (ADL-S GT1)", want: RendererIntelAlderLake},
		{in: "Mesa Intel(R) Graphics (RKL GT1)", want: RendererIntelRocketLake},
		{in: "Intel(R) Iris(TM) Graphics 6100", want: RendererIntelBroadwell},
		{in: "Intel(R) HD Graphics 530", want: RendererIntelSkyLake},
		{in: "AMD Radeon Pro 5300M OpenGL Engine", want: RendererAMDRadeonPro5xxx},
		{in: "AMD Radeon R9 M370X OpenGL Engine", want: RendererAMDRadeonR9M3xx},
		{in: "AMD Radeon R9 M470 OpenGL Engine", want: RendererAMDRadeonR9M4xx},
		{in: "AMD Radeon HD 7970 Series", want: RendererAMDRadeonHD7xxx},
		{in: "Radeon (TM) Pro Vega 20", want: RendererAMDRadeonProVegaxx},
		{in: "AMD Radeon RX 7900 XTX", want: RendererOther},
		{in: "NVIDIA GeForce GTX 775M OpenGL Engine", want: RendererOther},
		// Mobile families classify as Other under the desktop trim.
		{in: "Adreno (TM) 640", want: RendererOther},
		{in: "Mali-G76", want: RendererOther},
	}
	for _, c := range cases {
		if got := getRenderer(c.in); got != c.want {
			t.Errorf("getRenderer(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestGetDriverAndVersion(t *testing.T) {
	cases := []struct {
		version     string
		wantVersion DriverVersion
		vendor      Vendor
		wantDriver  Driver
	}{
		{vendor: VendorNVIDIA, version: "4.6.0 NVIDIA 545.29.06", wantDriver: DriverNVIDIA, wantVersion: DriverVer(545, 29, 0)},
		{vendor: VendorNVIDIA, version: "4.1 NVIDIA-16.0.12", wantDriver: DriverNVIDIA, wantVersion: DriverUnknownVersion},
		{vendor: VendorOther, version: "4.5 (Core Profile) Mesa 24.0.5", wantDriver: DriverMesa, wantVersion: DriverVer(24, 0, 0)},
		{vendor: VendorIntel, version: "4.6 (Core Profile) Mesa 22.3.6", wantDriver: DriverMesa, wantVersion: DriverVer(22, 3, 0)},
		{vendor: VendorOther, version: "3.3 Mesa 21.2.1", wantDriver: DriverMesa, wantVersion: DriverVer(21, 2, 0)},
		{vendor: VendorIntel, version: "4.1 INTEL-14.7.28", wantDriver: DriverIntel, wantVersion: DriverVer(14, 7, 28)},
		{vendor: VendorApple, version: "4.1 Metal - 90.5", wantDriver: DriverApple, wantVersion: DriverVer(90, 0, 0)},
		{vendor: VendorATI, version: "4.1 ATI-4.14.1", wantDriver: DriverUnknown, wantVersion: DriverUnknownVersion},
	}
	for _, c := range cases {
		driver, version := getDriverAndVersion(c.vendor, c.version)
		if driver != c.wantDriver || version != c.wantVersion {
			t.Errorf("getDriverAndVersion(%d, %q) = %d/%x, want %d/%x",
				c.vendor, c.version, driver, version, c.wantDriver, c.wantVersion)
		}
	}
}

func TestGetGLSLVersion(t *testing.T) {
	if got := getGLSLVersion("4.10"); got != Ver(4, 10) {
		t.Errorf("getGLSLVersion(4.10) = %v", got)
	}
	if got := getGLSLVersion("4.50 (Core Profile) Mesa 24.0.5"); got != Ver(4, 50) {
		t.Errorf("getGLSLVersion(mesa) = %v", got)
	}
	if got := getGLSLVersion("garbage"); got != InvalidVersion {
		t.Errorf("getGLSLVersion(garbage) = %v", got)
	}
}

func TestIsVirgl(t *testing.T) {
	if !isVirgl("virgl (Apple M1)") || isVirgl("Apple M1") {
		t.Error("virgl detection wrong")
	}
}
