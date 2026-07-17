// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The PDF object model: Object is the interface every primitive PDF element implements (emit); union is a tagged value
// holding the non-compound Name/String/Number/Boolean payloads; Array/OptionalArray/Dict are the compound objects built
// from unions. union carries a single active payload behind a type tag rather than a move-only variant, since Go's GC
// makes ownership transfer unnecessary; every emit path produces exact, deterministic PDF byte output.

package pdf

import (
	"strings"

	"github.com/richardwilkes/canvas/stream"
)

// Object is a primitive PDF element that can serialize itself.
type Object interface {
	// emit writes the object to the stream.
	emit(s stream.WStream)
}

// IndirectReference is a handle to an indirect object, identified by its object number. The zero value is invalid,
// since object numbers start at 1 and any non-positive value is treated as unset.
type IndirectReference struct {
	value int32
}

// IsValid reports whether the reference points at an object.
func (r IndirectReference) IsValid() bool { return r.value > 0 }

// unionType tags a union's active payload.
type unionType uint8

const (
	utDestroyed unionType = iota
	utInt
	utColorComponent
	utColorComponentF
	utBool
	utScalar
	utName        // an already-valid name, emitted without escaping
	utNameEscaped // a name that must be escaped on emit
	utByteString
	utTextString
	utObject
	utRef
)

// union is a tagged value holding one of the primitive PDF value payloads (int, bool, scalar, name, string, indirect
// reference, or nested object).
type union struct {
	obj Object
	s   string
	i   int32
	f   float32
	b   bool
	typ unionType
}

func unionInt(v int32) union            { return union{typ: utInt, i: v} }
func unionColorComponent(v uint8) union { return union{typ: utColorComponent, i: int32(v)} }
func unionColorComponentF(v float32) union {
	return union{typ: utColorComponentF, f: v}
}
func unionBool(v bool) union          { return union{typ: utBool, b: v} }
func unionScalar(v float32) union     { return union{typ: utScalar, f: v} }
func unionName(v string) union        { return union{typ: utName, s: v} }
func unionNameEscaped(v string) union { return union{typ: utNameEscaped, s: v} }
func unionByteString(v string) union  { return union{typ: utByteString, s: v} }
func unionTextString(v string) union  { return union{typ: utTextString, s: v} }
func unionObject(o Object) union      { return union{typ: utObject, obj: o} }

// unionRef wraps an indirect reference as a union payload.
func unionRef(r IndirectReference) union { return union{typ: utRef, i: r.value} }

// emit writes the union's active payload in its PDF-serialized form.
func (u *union) emit(s stream.WStream) {
	switch u.typ {
	case utInt:
		writeDecAsText(s, int(u.i))
	case utColorComponent:
		s.Write(appendColorComponent(nil, uint8(u.i)))
	case utColorComponentF:
		s.Write(appendColorComponentF(nil, u.f))
	case utBool:
		if u.b {
			writeText(s, "true")
		} else {
			writeText(s, "false")
		}
	case utScalar:
		s.Write(appendScalar(nil, u.f))
	case utName:
		writeText(s, "/")
		writeText(s, u.s)
	case utNameEscaped:
		writeText(s, "/")
		writeNameEscaped(s, u.s)
	case utByteString:
		writeByteString(s, u.s)
	case utTextString:
		writeTextString(s, u.s)
	case utObject:
		u.obj.emit(s)
	case utRef:
		writeDecAsText(s, int(u.i))
		writeText(s, " 0 R") // Generation number is always 0.
	default:
		panic("union.emit with bad type")
	}
}

// ---- string escaping --------------------------------------------------------------------------------

const kToEscape = "#/%()<>[]{}"

// writeNameEscaped writes name as a valid PDF name (not including the leading slash), escaping any byte that is not a
// printable non-special ASCII character as #XX.
func writeNameEscaped(o stream.WStream, name string) {
	for i := 0; i < len(name); i++ {
		v := name[i]
		if v < '!' || v > '~' || strings.IndexByte(kToEscape, v) >= 0 {
			o.Write([]byte{'#', hexUpper[v>>4], hexUpper[v&0xF]})
		} else {
			o.Write([]byte{v})
		}
	}
}

func writeLiteralByteString(s stream.WStream, cin string) {
	writeText(s, "(")
	for i := 0; i < len(cin); i++ {
		c := cin[i]
		if c < ' ' || '~' < c {
			s.Write([]byte{
				'\\',
				'0' | (c >> 6),
				'0' | ((c >> 3) & 0x07),
				'0' | (c & 0x07),
			})
		} else {
			if c == '\\' || c == '(' || c == ')' {
				writeText(s, "\\")
			}
			s.Write([]byte{c})
		}
	}
	writeText(s, ")")
}

func writeHexByteString(s stream.WStream, cin string) {
	writeText(s, "<")
	for i := 0; i < len(cin); i++ {
		c := cin[i]
		s.Write([]byte{hexUpper[c>>4], hexUpper[c&0xF]})
	}
	writeText(s, ">")
}

// writeOptimizedByteString chooses the shorter of the literal and hexadecimal encodings.
func writeOptimizedByteString(s stream.WStream, cin string, literalExtras int) {
	hexLength := 2 + 2*len(cin)
	literalLength := 2 + len(cin) + literalExtras
	if literalLength <= hexLength {
		writeLiteralByteString(s, cin)
	} else {
		writeHexByteString(s, cin)
	}
}

func writeByteString(s stream.WStream, cin string) {
	literalExtras := 0
	for i := 0; i < len(cin); i++ {
		c := cin[i]
		if c < ' ' || '~' < c {
			literalExtras += 3
		} else if c == '\\' || c == '(' || c == ')' {
			literalExtras++
		}
	}
	writeOptimizedByteString(s, cin, literalExtras)
}

// writeTextString writes a PDF text string: PDFDocEncoding literal/hex when the input is valid UTF-8 in that subset,
// else a UTF-16BE hex string with a byte-order mark. Invalid UTF-8 emits "<>".
func writeTextString(s stream.WStream, cin string) {
	b := []byte(cin)
	inputIsValidUTF8 := true
	inputIsPDFDocEncoding := true
	literalExtras := 0
	for i := 0; i < len(b); {
		unichar, next := nextUTF8(b, i)
		i = next
		if unichar < 0 {
			inputIsValidUTF8 = false
			break
		}
		// See Table D.2 (PDFDocEncoding Character Set) in the PDF3200_2008 spec.
		if (0x15 < unichar && unichar < 0x20) || 0x7E < unichar {
			inputIsPDFDocEncoding = false
			break
		}
		if unichar < ' ' || '~' < unichar {
			literalExtras += 3
		} else if unichar == '\\' || unichar == '(' || unichar == ')' {
			literalExtras++
		}
	}

	if !inputIsValidUTF8 {
		writeText(s, "<>")
		return
	}
	if inputIsPDFDocEncoding {
		writeOptimizedByteString(s, cin, literalExtras)
		return
	}
	writeText(s, "<FEFF")
	for i := 0; i < len(b); {
		unichar, next := nextUTF8(b, i)
		i = next
		writeUTF16beHex(s, unichar)
	}
	writeText(s, ">")
}

// ---- Array --------------------------------------------------------------------------------------------

// Array is a PDF array object: an ordered sequence of values.
type Array struct {
	values []union
}

// NewArray returns an empty PDF array.
func NewArray() *Array { return &Array{} }

// Size returns the number of elements in the array.
func (a *Array) Size() int { return len(a.values) }

func (a *Array) append(v union) { a.values = append(a.values, v) }

// AppendInt appends an integer value.
func (a *Array) AppendInt(v int32) { a.append(unionInt(v)) }

// AppendColorComponent appends a color component in [0,255].
func (a *Array) AppendColorComponent(v uint8) { a.append(unionColorComponent(v)) }

// AppendColorComponentF appends a floating-point color component.
func (a *Array) AppendColorComponentF(v float32) { a.append(unionColorComponentF(v)) }

// AppendBool appends a boolean value.
func (a *Array) AppendBool(v bool) { a.append(unionBool(v)) }

// AppendScalar appends a floating-point scalar value.
func (a *Array) AppendScalar(v float32) { a.append(unionScalar(v)) }

// AppendName appends a name value, escaping it as needed on emit.
func (a *Array) AppendName(v string) { a.append(unionNameEscaped(v)) }

// AppendByteString appends a byte string value.
func (a *Array) AppendByteString(v string) { a.append(unionByteString(v)) }

// AppendTextString appends a text string value.
func (a *Array) AppendTextString(v string) { a.append(unionTextString(v)) }

// AppendObject appends a nested object value.
func (a *Array) AppendObject(o Object) { a.append(unionObject(o)) }

// AppendRef appends an indirect reference value.
func (a *Array) AppendRef(r IndirectReference) { a.append(unionRef(r)) }

// emit writes the array in PDF array syntax: "[ v1 v2 ... ]".
func (a *Array) emit(s stream.WStream) {
	writeText(s, "[")
	for i := range a.values {
		a.values[i].emit(s)
		if i+1 < len(a.values) {
			writeText(s, " ")
		}
	}
	writeText(s, "]")
}

// OptionalArray is a PDF array that serializes as a bare value instead of a one-element array when it holds exactly one
// entry.
type OptionalArray struct {
	Array
}

// NewOptionalArray returns an empty optional array.
func NewOptionalArray() *OptionalArray { return &OptionalArray{} }

// emit writes the array, or its single element bare if it holds exactly one entry.
func (a *OptionalArray) emit(s stream.WStream) {
	if a.Size() == 1 {
		a.values[0].emit(s)
	} else {
		a.Array.emit(s)
	}
}

// ---- Dict ---------------------------------------------------------------------------------------------

type dictRecord struct {
	key   union
	value union
}

// Dict is a PDF dictionary object: an ordered sequence of key/value pairs.
type Dict struct {
	records []dictRecord
}

// NewDict returns an empty dictionary.
func NewDict() *Dict { return &Dict{} }

// NewTypedDict returns a dictionary whose Type entry is set to typ.
func NewTypedDict(typ string) *Dict {
	d := &Dict{}
	d.InsertName("Type", typ)
	return d
}

// Size returns the number of entries in the dictionary.
func (d *Dict) Size() int { return len(d.records) }

func (d *Dict) insert(key, value union) {
	d.records = append(d.records, dictRecord{key: key, value: value})
}

// InsertRef inserts an indirect reference value under key.
func (d *Dict) InsertRef(key string, ref IndirectReference) { d.insert(unionName(key), unionRef(ref)) }

// InsertObject inserts a nested object value under key.
func (d *Dict) InsertObject(key string, o Object) { d.insert(unionName(key), unionObject(o)) }

// InsertBool inserts a boolean value under key.
func (d *Dict) InsertBool(key string, value bool) { d.insert(unionName(key), unionBool(value)) }

// InsertInt inserts an integer value under key.
func (d *Dict) InsertInt(key string, value int32) { d.insert(unionName(key), unionInt(value)) }

// InsertScalar inserts a floating-point scalar value under key.
func (d *Dict) InsertScalar(key string, value float32) { d.insert(unionName(key), unionScalar(value)) }

// InsertColorComponentF inserts a floating-point color component value under key.
func (d *Dict) InsertColorComponentF(key string, value float32) {
	d.insert(unionName(key), unionColorComponentF(value))
}

// InsertName inserts a name value under key, escaping the name as needed on emit.
func (d *Dict) InsertName(key, name string) { d.insert(unionName(key), unionNameEscaped(name)) }

// InsertByteString inserts a byte string value under key.
func (d *Dict) InsertByteString(key, value string) { d.insert(unionName(key), unionByteString(value)) }

// InsertTextString inserts a text string value under key.
func (d *Dict) InsertTextString(key, value string) { d.insert(unionName(key), unionTextString(value)) }

// emit writes the dictionary in PDF dictionary syntax: "<< key value ... >>".
func (d *Dict) emit(s stream.WStream) {
	writeText(s, "<<")
	for i := range d.records {
		d.records[i].key.emit(s)
		writeText(s, " ")
		d.records[i].value.emit(s)
		if i+1 < len(d.records) {
			writeText(s, "\n")
		}
	}
	writeText(s, ">>")
}

// ---- variadic array constructors -----------------------------------------------------------------------

// makeArrayInts builds an array from a list of integers.
func makeArrayInts(vs ...int32) *Array {
	a := &Array{}
	a.values = make([]union, 0, len(vs))
	for _, v := range vs {
		a.AppendInt(v)
	}
	return a
}

// makeArrayScalars builds an array from a list of floating-point scalars.
func makeArrayScalars(vs ...float32) *Array {
	a := &Array{}
	a.values = make([]union, 0, len(vs))
	for _, v := range vs {
		a.AppendScalar(v)
	}
	return a
}

// emitToBytes writes obj as a self-contained byte slice, used by tests and for one-off object encoding that does not go
// through the document catalog.
func emitToBytes(o Object) []byte {
	s := stream.NewMemoryWStream()
	o.emit(s)
	return append([]byte(nil), s.Bytes()...)
}
