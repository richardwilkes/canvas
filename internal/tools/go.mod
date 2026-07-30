// A separate module so that the tools are not pulled into anyone's build of the canvas module. Nothing here is
// imported by the library, and the tools deliberately depend on nothing but the standard library.
module github.com/richardwilkes/canvas/internal/tools

go 1.26
