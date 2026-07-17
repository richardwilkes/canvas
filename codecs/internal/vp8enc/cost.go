// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Rate estimation: bit costs, level cost tables and mode cost tables. The fixed cost tables are embedded in
// costtables.go: the precomputed values are not all exactly the coding-tree costs (e.g. levelFixedCosts[9..10] omit the
// parity-bit cost, and fixedCostsUV uses complemented probabilities), and matching the reference encoder's rate
// estimates — hence its mode decisions — matters more than analytic purity.

package vp8enc

// bitCost returns the cost, in 1/256th bits, of coding 'bit' with probability proba.
func bitCost(bit bool, proba uint8) int {
	if bit {
		return int(entropyCost[255-proba])
	}
	return int(entropyCost[proba])
}

//------------------------------------------------------------------------------
// Probability state

type (
	statsArray [numCtx][numProbas]uint32 // 16-bit "ones" count + 16-bit total count
	costArray  [numCtx][maxVariableLevel + 1]uint16
)

// encProba holds the token probabilities, recorded statistics and derived level cost tables.
type encProba struct {
	remappedCosts [numTypes][16]*costArray // per coefficient position, indexed via encBands
	nbSkip        int
	stats         [numTypes][numBands]statsArray
	levelCost     [numTypes][numBands]costArray
	coeffs        [numTypes][numBands][numCtx][numProbas]uint8
	dirty         bool
	useSkipProba  bool
	skipProba     uint8
}

func (p *encProba) initDefaults() {
	p.useSkipProba = false
	p.coeffs = coeffsProba0
	p.dirty = true
}

// recordStats increments the (bit, total) counters, halving both on overflow; returns bit.
func recordStats(bit bool, stats *uint32) bool {
	s := *stats
	if s >= 0xfffe0000 { // an overflow is inbound: divide the stats by 2
		s = ((s + 1) >> 1) & 0x7fff7fff
	}
	s += 0x00010000
	if bit {
		s++
	}
	*stats = s
	return bit
}

// variableLevelCost returns the context-probability part of coding level (1..maxVariableLevel).
func variableLevelCost(level int, probas *[numProbas]uint8) int {
	pattern := int(levelCodes[level-1][0])
	bits := int(levelCodes[level-1][1])
	cost := 0
	for i := 2; pattern != 0; i++ {
		if pattern&1 != 0 {
			cost += bitCost(bits&1 != 0, probas[i])
		}
		bits >>= 1
		pattern >>= 1
	}
	return cost
}

// calculateLevelCosts refreshes the level cost tables from the current probabilities.
func (p *encProba) calculateLevelCosts() {
	if !p.dirty {
		return
	}
	for ctype := 0; ctype < numTypes; ctype++ {
		for band := 0; band < numBands; band++ {
			for ctx := 0; ctx < numCtx; ctx++ {
				probas := &p.coeffs[ctype][band][ctx]
				table := &p.levelCost[ctype][band][ctx]
				cost0 := 0
				if ctx > 0 {
					cost0 = bitCost(true, probas[0])
				}
				costBase := bitCost(true, probas[1]) + cost0
				table[0] = uint16(bitCost(false, probas[1]) + cost0)
				for v := 1; v <= maxVariableLevel; v++ {
					table[v] = uint16(costBase + variableLevelCost(v, probas))
				}
			}
		}
		for n := 0; n < 16; n++ { // replicate the bands
			p.remappedCosts[ctype][n] = &p.levelCost[ctype][encBands[n]]
		}
	}
	p.dirty = false
}

// levelCost returns the full cost of coding the absolute level with the given cost table row.
func levelCost(table *[maxVariableLevel + 1]uint16, level int) int {
	v := level
	if v > maxVariableLevel {
		v = maxVariableLevel
	}
	return int(levelFixedCosts[level]) + int(table[v])
}

//------------------------------------------------------------------------------
// Residuals

// residual describes one 16-coefficient block to code or cost.
type residual struct {
	coeffs    *[16]int16
	first     int // 0, or 1 for the AC coefficients of an intra16 macroblock
	last      int // last non-zero coefficient, or -1
	coeffType int
}

func (r *residual) init(first, coeffType int) {
	r.first = first
	r.coeffType = coeffType
}

func (r *residual) setCoeffs(coeffs *[16]int16) {
	r.last = -1
	for n := 15; n >= 0; n-- {
		if coeffs[n] != 0 {
			r.last = n
			break
		}
	}
	r.coeffs = coeffs
}

// getResidualCost estimates the coding cost of the residual with the current probabilities.
func (p *encProba) getResidualCost(ctx0 int, res *residual) int {
	n := res.first
	// The probability row for n is prob[encBands[n]], equivalent to prob[n] for n == 0 or 1.
	p0 := p.coeffs[res.coeffType][n][ctx0][0]
	costs := &p.remappedCosts[res.coeffType]
	t := &costs[n][ctx0]
	// bitCost(true, p0) is already incorporated in t[] for ctx != 0 (as required by the syntax); for ctx0 == 0 it must
	// be added here.
	cost := 0
	if ctx0 == 0 {
		cost = bitCost(true, p0)
	}
	if res.last < 0 {
		return bitCost(false, p0)
	}
	for ; n < res.last; n++ {
		v := int(res.coeffs[n])
		if v < 0 {
			v = -v
		}
		ctx := v
		if ctx > 2 {
			ctx = 2
		}
		cost += levelCost(t, v)
		t = &costs[n+1][ctx]
	}
	// The last coefficient is always non-zero.
	v := int(res.coeffs[n])
	if v < 0 {
		v = -v
	}
	cost += levelCost(t, v)
	if n < 15 {
		b := encBands[n+1]
		ctx := 2
		if v == 1 {
			ctx = 1
		}
		lastP0 := p.coeffs[res.coeffType][b][ctx][0]
		cost += bitCost(false, lastP0)
	}
	return cost
}

// recordCoeffs simulates coding the residual, recording branch statistics; returns whether the block has any non-zero
// coefficient.
func (p *encProba) recordCoeffs(ctx int, res *residual) int {
	n := res.first
	s := &p.stats[res.coeffType][n][ctx] // stats[encBands[n]] is equivalent for n == 0 or 1
	if res.last < 0 {
		recordStats(false, &s[0])
		return 0
	}
	for n <= res.last {
		recordStats(true, &s[0])
		var v int
		for {
			v = int(res.coeffs[n])
			n++
			if v != 0 {
				break
			}
			recordStats(false, &s[1])
			s = &p.stats[res.coeffType][encBands[n]][0]
		}
		recordStats(true, &s[1])
		if !recordStats(v > 1 || v < -1, &s[2]) { // false means v == -1 or 1
			s = &p.stats[res.coeffType][encBands[n]][1]
		} else {
			if v < 0 {
				v = -v
			}
			if v > maxVariableLevel {
				v = maxVariableLevel
			}
			bits := int(levelCodes[v-1][1])
			pattern := int(levelCodes[v-1][0])
			for i := 0; ; i++ {
				pattern >>= 1
				if pattern == 0 {
					break
				}
				mask := 2 << uint(i)
				if pattern&1 != 0 {
					recordStats(bits&mask != 0, &s[3+i])
				}
			}
			s = &p.stats[res.coeffType][encBands[n]][2]
		}
	}
	if n < 16 {
		recordStats(false, &s[0])
	}
	return 1
}

// putCoeffs codes the residual into the boolean writer, returning whether the block has any non-zero coefficient.
func (p *encProba) putCoeffs(bw *bitWriter, ctx int, res *residual) int {
	n := res.first
	probRow := &p.coeffs[res.coeffType][n][ctx] // prob[encBands[n]], equivalent for n == 0 or 1
	if !bw.putBit(res.last >= 0, probRow[0]) {
		return 0
	}
	for n < 16 {
		c := int(res.coeffs[n])
		n++
		sign := c < 0
		v := c
		if sign {
			v = -c
		}
		if !bw.putBit(v != 0, probRow[1]) {
			probRow = &p.coeffs[res.coeffType][encBands[n]][0]
			continue
		}
		if !bw.putBit(v > 1, probRow[2]) {
			probRow = &p.coeffs[res.coeffType][encBands[n]][1]
		} else {
			switch {
			case !bw.putBit(v > 4, probRow[3]):
				if bw.putBit(v != 2, probRow[4]) {
					bw.putBit(v == 4, probRow[5])
				}
			case !bw.putBit(v > 10, probRow[6]):
				if !bw.putBit(v > 6, probRow[7]) {
					bw.putBit(v == 6, 159)
				} else {
					bw.putBit(v >= 9, 165)
					bw.putBit(v&1 == 0, 145)
				}
			default:
				var mask int
				var tab []uint8
				switch {
				case v < 3+(8<<1): // cat3 (3 bits)
					bw.putBit(false, probRow[8])
					bw.putBit(false, probRow[9])
					v -= 3 + (8 << 0)
					mask = 1 << 2
					tab = cat3[:]
				case v < 3+(8<<2): // cat4 (4 bits)
					bw.putBit(false, probRow[8])
					bw.putBit(true, probRow[9])
					v -= 3 + (8 << 1)
					mask = 1 << 3
					tab = cat4[:]
				case v < 3+(8<<3): // cat5 (5 bits)
					bw.putBit(true, probRow[8])
					bw.putBit(false, probRow[10])
					v -= 3 + (8 << 2)
					mask = 1 << 4
					tab = cat5[:]
				default: // cat6 (11 bits)
					bw.putBit(true, probRow[8])
					bw.putBit(true, probRow[10])
					v -= 3 + (8 << 3)
					mask = 1 << 10
					tab = cat6[:]
				}
				for i := 0; mask != 0; i++ {
					bw.putBit(v&mask != 0, tab[i])
					mask >>= 1
				}
			}
			probRow = &p.coeffs[res.coeffType][encBands[n]][2]
		}
		bw.putBitUniform(sign)
		if n == 16 || !bw.putBit(n <= res.last, probRow[0]) {
			return 1 // EOB
		}
	}
	return 1
}

//------------------------------------------------------------------------------
// Probability finalization

const skipProbaThreshold = 250 // value below which using skip_proba is OK

func calcSkipProba(nb, total int) uint8 {
	if total > 0 {
		return uint8((total - nb) * 255 / total)
	}
	return 255
}

// finalizeSkipProba computes the skip probability and whether to use it, returning the header bit-cost of the choice.
func (p *encProba) finalizeSkipProba(nbMBs int) int {
	p.skipProba = calcSkipProba(p.nbSkip, nbMBs)
	p.useSkipProba = p.skipProba < skipProbaThreshold
	size := 256 // 'use skip proba' bit
	if p.useSkipProba {
		size += p.nbSkip*bitCost(true, p.skipProba) +
			(nbMBs-p.nbSkip)*bitCost(false, p.skipProba)
		size += 8 * 256 // cost of signaling skipProba itself
	}
	return size
}

func calcTokenProba(nb, total int) uint8 {
	if nb == 0 {
		return 255
	}
	return uint8(255 - nb*255/total)
}

// branchCost is the cost of coding nb ones and total-nb zeros with probability proba.
func branchCost(nb, total int, proba uint8) int {
	return nb*bitCost(true, proba) + (total-nb)*bitCost(false, proba)
}

// finalizeTokenProbas updates the token probabilities from the recorded statistics, choosing per branch whether
// signaling the update is worth its cost. Returns the total header bit-cost.
func (p *encProba) finalizeTokenProbas() int {
	hasChanged := false
	size := 0
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for i := 0; i < numProbas; i++ {
					stats := p.stats[t][b][c][i]
					nb := int(stats & 0xffff)
					total := int((stats >> 16) & 0xffff)
					updateProba := coeffsUpdateProba[t][b][c][i]
					oldP := coeffsProba0[t][b][c][i]
					newP := calcTokenProba(nb, total)
					oldCost := branchCost(nb, total, oldP) + bitCost(false, updateProba)
					newCost := branchCost(nb, total, newP) + bitCost(true, updateProba) + 8*256
					useNewP := oldCost > newCost
					size += bitCost(useNewP, updateProba)
					if useNewP { // only use probas that seem meaningful enough
						p.coeffs[t][b][c][i] = newP
						hasChanged = hasChanged || newP != oldP
						size += 8 * 256
					} else {
						p.coeffs[t][b][c][i] = oldP
					}
				}
			}
		}
	}
	p.dirty = hasChanged
	return size
}

func (p *encProba) resetTokenStats() {
	p.stats = [numTypes][numBands]statsArray{}
}

// writeProbas emits the token probability updates and the skip probability.
func (p *encProba) writeProbas(bw *bitWriter) {
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for i := 0; i < numProbas; i++ {
					p0 := p.coeffs[t][b][c][i]
					update := p0 != coeffsProba0[t][b][c][i]
					if bw.putBit(update, coeffsUpdateProba[t][b][c][i]) {
						bw.putBits(uint32(p0), 8)
					}
				}
			}
		}
	}
	if bw.putBitUniform(p.useSkipProba) {
		bw.putBits(uint32(p.skipProba), 8)
	}
}
