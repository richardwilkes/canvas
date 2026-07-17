// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The intersection-time coincidence tracking surface: the list of coincident span pairs (coincidentSpans) plus the
// pieces the intersection-adding walker needs — add (record a coincident run), isEmpty, markCollapsed, releaseDeleted,
// fixUp, and the Ordered comparator. The coincidence-resolution machinery (addExpanded, addMissing, mark, apply,
// expand, findOverlaps, …) that runs in the walking phase arrives with its own slice; the fields are present but those
// methods are not.

package pathops

import "github.com/richardwilkes/canvas/geom"

// coincidentSpans is one run of two segments that coincide over a t range. The start/end pt-t pairs always name the
// span's primary pt-t.
type coincidentSpans struct {
	next         *coincidentSpans
	coinPtTStart *opPtT
	coinPtTEnd   *opPtT
	oppPtTStart  *opPtT
	oppPtTEnd    *opPtT
}

// setCoinPtTStart sets the run's coincident-segment start pt-t and flags it coincident.
func (cs *coincidentSpans) setCoinPtTStart(ptT *opPtT) {
	cs.coinPtTStart = ptT
	ptT.setCoincident()
}

// setCoinPtTEnd sets the run's coincident-segment end pt-t and flags it coincident.
func (cs *coincidentSpans) setCoinPtTEnd(ptT *opPtT) {
	cs.coinPtTEnd = ptT
	ptT.setCoincident()
}

// setOppPtTStart sets the run's opposite-segment start pt-t and flags it coincident.
func (cs *coincidentSpans) setOppPtTStart(ptT *opPtT) {
	cs.oppPtTStart = ptT
	ptT.setCoincident()
}

// setOppPtTEnd sets the run's opposite-segment end pt-t and flags it coincident.
func (cs *coincidentSpans) setOppPtTEnd(ptT *opPtT) {
	cs.oppPtTEnd = ptT
	ptT.setCoincident()
}

// setStarts sets both segments' start pt-t for this run.
func (cs *coincidentSpans) setStarts(coinPtTStart, oppPtTStart *opPtT) {
	cs.setCoinPtTStart(coinPtTStart)
	cs.setOppPtTStart(oppPtTStart)
}

// setEnds sets both segments' end pt-t for this run.
func (cs *coincidentSpans) setEnds(coinPtTEnd, oppPtTEnd *opPtT) {
	cs.setCoinPtTEnd(coinPtTEnd)
	cs.setOppPtTEnd(oppPtTEnd)
}

// set initializes the run's next link and both segments' start/end pt-t pairs.
func (cs *coincidentSpans) set(next *coincidentSpans, coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd *opPtT) {
	cs.next = next
	cs.setStarts(coinPtTStart, oppPtTStart)
	cs.setEnds(coinPtTEnd, oppPtTEnd)
}

// flipped reports whether the opposite run reverses this run's t direction.
func (cs *coincidentSpans) flipped() bool { return cs.oppPtTStart.t > cs.oppPtTEnd.t }

// collapsed reports whether this run's start and end have collapsed onto the same point (one end is test and the other
// end's pt-t loop now contains test).
func (cs *coincidentSpans) collapsed(test *opPtT) bool {
	return (cs.coinPtTStart == test && cs.coinPtTEnd.containsPtT(test)) ||
		(cs.coinPtTEnd == test && cs.coinPtTStart.containsPtT(test)) ||
		(cs.oppPtTStart == test && cs.oppPtTEnd.containsPtT(test)) ||
		(cs.oppPtTEnd == test && cs.oppPtTStart.containsPtT(test))
}

// opCoincidence is the collection of coincident span runs discovered during the intersection phase. top holds runs
// migrated by the (deferred) expansion machinery; the walker only ever pushes onto head, but the traversal helpers scan
// both lists.
type opCoincidence struct {
	head        *coincidentSpans
	top         *coincidentSpans
	globalState *opGlobalState
}

// newOpCoincidence creates a coincidence tracker and registers it on the global state.
func newOpCoincidence(globalState *opGlobalState) *opCoincidence {
	co := &opCoincidence{globalState: globalState}
	globalState.setCoincidence(co)
	return co
}

// isEmpty reports whether no coincident runs have been recorded.
func (co *opCoincidence) isEmpty() bool { return co.head == nil && co.top == nil }

// add records a coincident run, ordering the two segments canonically (the caller may pass them in either order) and
// tracking the front-of-loop pt-t for each end.
func (co *opCoincidence) add(coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd *opPtT) {
	if !coincidenceOrdered(coinPtTStart, oppPtTStart) {
		if oppPtTStart.t < oppPtTEnd.t {
			co.add(oppPtTStart, oppPtTEnd, coinPtTStart, coinPtTEnd)
		} else {
			co.add(oppPtTEnd, oppPtTStart, coinPtTEnd, coinPtTStart)
		}
		return
	}
	// choose the ptT at the front of the list to track
	coinPtTStart = coinPtTStart.span.ptTPtr()
	coinPtTEnd = coinPtTEnd.span.ptTPtr()
	oppPtTStart = oppPtTStart.span.ptTPtr()
	oppPtTEnd = oppPtTEnd.span.ptTPtr()
	coinRec := &coincidentSpans{}
	coinRec.set(co.head, coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd)
	co.head = coinRec
}

// release unlinks remove from the coin list headed by coin. Returns true if remove was found.
func (co *opCoincidence) release(coin, remove *coincidentSpans) bool {
	head := coin
	var prev *coincidentSpans
	for coin != nil {
		next := coin.next
		if coin == remove {
			switch {
			case prev != nil:
				prev.next = next
			case head == co.head:
				co.head = next
			default:
				co.top = next
			}
			break
		}
		prev = coin
		coin = next
	}
	return coin != nil
}

// releaseDeleted drops every run whose tracked start pt-t was deleted, across both lists.
func (co *opCoincidence) releaseDeleted() {
	co.releaseDeletedList(co.head)
	co.releaseDeletedList(co.top)
}

func (co *opCoincidence) releaseDeletedList(coin *coincidentSpans) {
	if coin == nil {
		return
	}
	head := coin
	var prev *coincidentSpans
	for coin != nil {
		next := coin.next
		if coin.coinPtTStart.deleted {
			switch {
			case prev != nil:
				prev.next = next
			case head == co.head:
				co.head = next
			default:
				co.top = next
			}
		} else {
			prev = coin
		}
		coin = next
	}
}

// fixUp repoints (or drops) every run that named the deleted pt-t at the kept pt-t; called after a span is released.
func (co *opCoincidence) fixUp(deleted, kept *opPtT) {
	if co.head != nil {
		co.fixUpList(co.head, deleted, kept)
	}
	if co.top != nil {
		co.fixUpList(co.top, deleted, kept)
	}
}

func (co *opCoincidence) fixUpList(coin *coincidentSpans, deleted, kept *opPtT) {
	head := coin
	for coin != nil {
		if coin.coinPtTStart == deleted {
			if coin.coinPtTEnd.span == kept.span {
				co.release(head, coin)
				coin = coin.next
				continue
			}
			coin.setCoinPtTStart(kept)
		}
		if coin.coinPtTEnd == deleted {
			if coin.coinPtTStart.span == kept.span {
				co.release(head, coin)
				coin = coin.next
				continue
			}
			coin.setCoinPtTEnd(kept)
		}
		if coin.oppPtTStart == deleted {
			if coin.oppPtTEnd.span == kept.span {
				co.release(head, coin)
				coin = coin.next
				continue
			}
			coin.setOppPtTStart(kept)
		}
		if coin.oppPtTEnd == deleted {
			if coin.oppPtTStart.span == kept.span {
				co.release(head, coin)
				coin = coin.next
				continue
			}
			coin.setOppPtTEnd(kept)
		}
		coin = coin.next
	}
}

// markCollapsed marks, across both lists, every run that has collapsed onto test — marking the involved segments done
// and releasing the run.
func (co *opCoincidence) markCollapsed(test *opPtT) {
	co.markCollapsedList(co.head, test)
	co.markCollapsedList(co.top, test)
}

func (co *opCoincidence) markCollapsedList(coin *coincidentSpans, test *opPtT) {
	head := coin
	for coin != nil {
		if coin.collapsed(test) {
			if zeroOrOne(coin.coinPtTStart.t) && zeroOrOne(coin.coinPtTEnd.t) {
				coin.coinPtTStart.segment().markAllDone()
			}
			if zeroOrOne(coin.oppPtTStart.t) && zeroOrOne(coin.oppPtTEnd.t) {
				coin.oppPtTStart.segment().markAllDone()
			}
			co.release(head, coin)
		}
		coin = coin.next
	}
}

// coincidenceOrdered reports whether the segments owning coinPtTStart and oppPtTStart are already in canonical order
// (see coincidenceOrderedSeg).
func coincidenceOrdered(coinPtTStart, oppPtTStart *opPtT) bool {
	return coincidenceOrderedSeg(coinPtTStart.segment(), oppPtTStart.segment())
}

// coincidenceOrderedSeg orders two segments by verb, then lexicographically by their raw point coordinate stream.
func coincidenceOrderedSeg(coinSeg, oppSeg *opSegment) bool {
	if coinSeg.verb < oppSeg.verb {
		return true
	}
	if coinSeg.verb > oppSeg.verb {
		return false
	}
	count := (opVerbToPoints(coinSeg.verb) + 1) * 2
	for index := 0; index < count; index++ {
		c := coordAt(coinSeg.pts, index)
		o := coordAt(oppSeg.pts, index)
		if c < o {
			return true
		}
		if c > o {
			return false
		}
	}
	return true
}

// coordAt indexes the flat (x, y, x, y, …) coordinate stream of a point slice.
func coordAt(pts []geom.Point, index int) float32 {
	p := pts[index/2]
	if index%2 == 0 {
		return p.X
	}
	return p.Y
}
