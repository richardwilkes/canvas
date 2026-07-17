// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Driver-identity detection: vendor, renderer, driver, and driver-version parsing from the
// GL_VENDOR/GL_RENDERER/GL_VERSION strings. Per the desktop trim, the ANGLE, WebGL, and Chrome-command-buffer detection
// paths are dropped along with the ES-only vendor branches of the driver-version parser; the renderer table keeps the
// families that occur on desktop GL (Intel generations, AMD Radeon, Apple, NVIDIA Tegra excluded, Mesa llvmpipe) and
// classifies everything else as Other. The virgl detection is kept — VirGL guests expose desktop GL.

package gl

import "strings"

// DriverVersion packs a driver version as (major << 32) | (minor << 16) | point.
type DriverVersion uint64

// DriverVer packs major/minor/point into a DriverVersion.
func DriverVer(major, minor, point uint64) DriverVersion {
	return DriverVersion(major<<32 | (minor&0xFFFF)<<16 | (point & 0xFFFF))
}

// DriverUnknownVersion is the sentinel for an unrecognized driver version.
const DriverUnknownVersion DriverVersion = 0

// Vendor identifies the GL vendor.
type Vendor int32

// Vendor values.
const (
	VendorARM Vendor = iota
	VendorGoogle
	VendorImagination
	VendorIntel
	VendorQualcomm
	VendorNVIDIA
	VendorATI
	VendorApple
	VendorOther
)

// Renderer identifies the GL renderer, trimmed to the values reachable on desktop GL. The mobile families (Adreno,
// Mali, PowerVR, Tegra) only appear behind ES contexts, which assembly rejects.
type Renderer int32

// Renderer values.
const (
	// Intel GPU families, ordered by generation.
	RendererIntelSandyBridge Renderer = iota // 6th gen
	RendererIntelIvyBridge                   // 7th gen
	RendererIntelValleyView                  // aka BayTrail
	RendererIntelHaswell
	RendererIntelCherryView // aka Braswell, 8th gen
	RendererIntelBroadwell
	RendererIntelApolloLake // 9th gen
	RendererIntelSkyLake
	RendererIntelGeminiLake
	RendererIntelKabyLake
	RendererIntelCoffeeLake
	RendererIntelIceLake    // 11th gen
	RendererIntelRocketLake // 12th gen
	RendererIntelTigerLake
	RendererIntelAlderLake

	RendererGalliumLLVM

	RendererAMDRadeonHD7xxx    // AMD Radeon HD 7000 Series
	RendererAMDRadeonR9M3xx    // AMD Radeon R9 M300 Series
	RendererAMDRadeonR9M4xx    // AMD Radeon R9 M400 Series
	RendererAMDRadeonPro5xxx   // AMD Radeon Pro 5000 Series
	RendererAMDRadeonProVegaxx // AMD Radeon Pro Vega

	RendererApple

	RendererOther
)

// Driver identifies the GL driver, trimmed to the drivers detectable on desktop GL.
type Driver int32

// Driver values.
const (
	DriverMesa Driver = iota
	DriverNVIDIA
	DriverIntel
	DriverApple
	DriverUnknown
)

// DriverInfo holds the detected driver identity (minus the ANGLE/WebGL/command-buffer fields, per the trim).
type DriverInfo struct {
	// RendererString retains the raw GL_RENDERER string; the live-test harness keys non-hardware-GL detection on it.
	RendererString string
	DriverVersion  DriverVersion
	Standard       Standard
	Version        Version
	GLSLVersion    Version
	Vendor         Vendor
	Renderer       Renderer
	Driver         Driver
	// IsRunningOverVirgl: running over the virgl guest driver.
	IsRunningOverVirgl bool
}

// getGLSLVersion parses a GLSL version string (desktop pattern only; the ES patterns are unreachable).
func getGLSLVersion(versionString string) Version {
	if v, n := scanFormat(versionString, "%d.%d"); n == 2 {
		return Ver(uint32(v[0]), uint32(v[1]))
	}
	return InvalidVersion
}

// getVendor maps a GL_VENDOR string to a Vendor.
func getVendor(vendorString string) Vendor {
	switch {
	case vendorString == "ARM":
		return VendorARM
	case vendorString == "Google Inc.":
		return VendorGoogle
	case vendorString == "Imagination Technologies":
		return VendorImagination
	case strings.HasPrefix(vendorString, "Intel ") || vendorString == "Intel":
		return VendorIntel
	case vendorString == "Qualcomm" || vendorString == "freedreno":
		return VendorQualcomm
	case vendorString == "NVIDIA Corporation":
		return VendorNVIDIA
	case vendorString == "ATI Technologies Inc.":
		return VendorATI
	case vendorString == "Apple":
		return VendorApple
	default:
		return VendorOther
	}
}

// getRenderer maps a GL_RENDERER string to a Renderer, desktop-reachable subset only. The Tegra/PowerVR/Adreno/Mali
// blocks are dropped (mobile trim); anything they would have matched classifies as RendererOther.
func getRenderer(rendererString string) Renderer {
	if idx := strings.Index(rendererString, "Intel"); idx >= 0 {
		intelString := rendererString[idx:]
		// These generic strings seem to always come from Haswell: Iris 5100 or Iris Pro 5200.
		if intelString == "Intel Iris OpenGL Engine" || intelString == "Intel Iris Pro OpenGL Engine" {
			return RendererIntelHaswell
		}
		if strings.Contains(intelString, "Sandybridge") {
			return RendererIntelSandyBridge
		}
		if strings.Contains(intelString, "Bay Trail") {
			return RendererIntelValleyView
		}
		// In Mesa, 'RKL' can be followed by 'Graphics', same for 'TGL' and 'ADL'.
		if strings.Contains(intelString, "RKL") {
			return RendererIntelRocketLake
		}
		if strings.Contains(intelString, "TGL") {
			return RendererIntelTigerLake
		}
		// For Windows on ADL-S devices, 'AlderLake-S' might be followed by 'Intel(R)'.
		if strings.Contains(intelString, "ADL") || strings.Contains(intelString, "AlderLake") {
			return RendererIntelAlderLake
		}
		// For Windows on TGL or other ADL devices, we might only get 'Xe' from the string. Since they are both 12th
		// gen, use TigerLake to cover both.
		if strings.Contains(intelString, "Xe") {
			return RendererIntelTigerLake
		}
		// There are many possible intervening strings ('Intel(R)', 'Iris', '(R)', '(TM)', 'Pro', 'Plus', 'HD', 'UHD',
		// ...) but in all cases we end with 'Graphics ', an optional 'P', and a number.
		if gfxIdx := strings.Index(intelString, "Graphics"); gfxIdx >= 0 {
			intelGfxString := intelString[gfxIdx:]
			var intelNumber int
			found := false
			if v, n := scanFormat(intelGfxString, "Graphics %d"); n == 1 {
				intelNumber = v[0]
				found = true
			} else if v, n = scanFormat(intelGfxString, "Graphics P%d"); n == 1 {
				intelNumber = v[0]
				found = true
			}
			if found {
				switch {
				case intelNumber == 2000 || intelNumber == 3000:
					return RendererIntelSandyBridge
				case intelNumber == 2500 || intelNumber == 4000:
					return RendererIntelIvyBridge
				case intelNumber >= 4200 && intelNumber <= 5200:
					return RendererIntelHaswell
				case intelNumber >= 400 && intelNumber <= 405:
					return RendererIntelCherryView
				case intelNumber >= 5300 && intelNumber <= 6300:
					return RendererIntelBroadwell
				case intelNumber >= 500 && intelNumber <= 505:
					return RendererIntelApolloLake
				case intelNumber >= 510 && intelNumber <= 580:
					return RendererIntelSkyLake
				case intelNumber >= 600 && intelNumber <= 605:
					return RendererIntelGeminiLake
				// 610 and 630 are reused from KabyLake to CoffeeLake. The CoffeeLake variants are "UHD Graphics", while
				// the KabyLake ones are "HD Graphics".
				case intelNumber == 610 || intelNumber == 630:
					if strings.Contains(intelString, "UHD") {
						return RendererIntelCoffeeLake
					}
					return RendererIntelKabyLake
				case intelNumber >= 610 && intelNumber <= 650:
					return RendererIntelKabyLake
				case intelNumber == 655:
					return RendererIntelCoffeeLake
				// 710/730/750/770 are all 12th gen UHD Graphics, but it's hard to distinguish among RKL, TGL and ADL.
				// Use TigerLake to cover all.
				case intelNumber >= 710 && intelNumber <= 770:
					return RendererIntelTigerLake
				case intelNumber >= 910 && intelNumber <= 950:
					return RendererIntelIceLake
				}
			}
		}
	}

	// The AMD string can have a somewhat arbitrary preamble (see skbug.com/40038435).
	if idx := strings.Index(rendererString, "Radeon "); idx >= 0 {
		amdString := rendererString[idx+len("Radeon "):]
		// Sometimes there is a (TM) and sometimes not.
		amdString = strings.TrimPrefix(amdString, "(TM) ")

		if v, n := scanFormat(amdString, "R9 M3%c%c"); n == 2 && isDigit(byte(v[0])) && isDigit(byte(v[1])) {
			return RendererAMDRadeonR9M3xx
		}
		if v, n := scanFormat(amdString, "R9 M4%c%c"); n == 2 && isDigit(byte(v[0])) && isDigit(byte(v[1])) {
			return RendererAMDRadeonR9M4xx
		}
		if v, n := scanFormat(amdString, "HD 7%c%c%c Series"); n == 3 && isDigit(byte(v[0])) &&
			isDigit(byte(v[1])) && isDigit(byte(v[2])) {
			return RendererAMDRadeonHD7xxx
		}
		if v, n := scanFormat(amdString, "Pro 5%c%c%c"); n == 3 && isDigit(byte(v[0])) &&
			isDigit(byte(v[1])) && isDigit(byte(v[2])) {
			return RendererAMDRadeonPro5xxx
		}
		if _, n := scanFormat(amdString, "Pro Vega %d"); n == 1 {
			return RendererAMDRadeonProVegaxx
		}
	}

	if strings.Contains(rendererString, "llvmpipe") {
		return RendererGalliumLLVM
	}
	if strings.HasPrefix(rendererString, "Apple") {
		return RendererApple
	}
	return RendererOther
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// isVirgl reports whether rendererString identifies the virgl guest driver.
func isVirgl(rendererString string) bool { return strings.Contains(rendererString, "virgl") }

// getDriverAndVersion parses the driver and its version from a GL_VERSION string, desktop-GL branches only. The ES
// vendor branches (Qualcomm, Imagination, ARM, Android emulator) are dropped.
func getDriverAndVersion(vendor Vendor, versionString string) (Driver, DriverVersion) {
	driver := DriverUnknown
	driverVersion := DriverUnknownVersion

	if vendor == VendorNVIDIA {
		driver = DriverNVIDIA
		// Some older NVIDIA drivers don't report the driver version.
		if v, n := scanFormat(versionString, "%d.%d.%d NVIDIA %d.%d"); n == 5 {
			driverVersion = DriverVer(uint64(v[3]), uint64(v[4]), 0)
		}
		return driver, driverVersion
	}
	if v, n := scanFormat(versionString, "%d.%d Mesa %d.%d"); n == 4 {
		return DriverMesa, DriverVer(uint64(v[2]), uint64(v[3]), 0)
	}
	if v, n := scanFormat(versionString, "%d.%d (Core Profile) Mesa %d.%d"); n == 4 {
		return DriverMesa, DriverVer(uint64(v[2]), uint64(v[3]), 0)
	}

	switch vendor {
	case VendorIntel:
		// We presume we're on the Intel driver since it hasn't identified itself as Mesa. This is how the macOS version
		// strings are structured; it might differ on other OSes.
		driver = DriverIntel
		if v, n := scanFormat(versionString, "%d.%d INTEL-%d.%d.%d"); n == 5 {
			driverVersion = DriverVer(uint64(v[2]), uint64(v[3]), uint64(v[4]))
		}
	case VendorApple:
		// There doesn't appear to be a minor version.
		if v, n := scanFormat(versionString, "%d.%d Metal - %d"); n == 3 {
			driver = DriverApple
			driverVersion = DriverVer(uint64(v[2]), 0, 0)
		}
	}
	return driver, driverVersion
}

// GetDriverInfo detects the driver identity from iface, desktop-trimmed.
func GetDriverInfo(iface *Interface) DriverInfo {
	info := DriverInfo{
		Vendor:   VendorOther,
		Renderer: RendererOther,
		Driver:   DriverUnknown,
	}
	if iface == nil {
		return info
	}
	info.Standard = iface.Standard
	f := &iface.Functions
	version := f.GetStringGo(VERSION)
	slversion := f.GetStringGo(SHADING_LANGUAGE_VERSION)
	renderer := f.GetStringGo(RENDERER)
	vendor := f.GetStringGo(VENDOR)

	info.Version = GetVersionFromString(version)
	info.GLSLVersion = getGLSLVersion(slversion)
	info.Vendor = getVendor(vendor)
	info.Renderer = getRenderer(renderer)
	info.RendererString = renderer
	info.Driver, info.DriverVersion = getDriverAndVersion(info.Vendor, version)
	info.IsRunningOverVirgl = isVirgl(renderer)
	return info
}
