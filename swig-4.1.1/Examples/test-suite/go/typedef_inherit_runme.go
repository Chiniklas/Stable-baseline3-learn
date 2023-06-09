package main

import "swigtests/typedef_inherit"

func main() {
	a := typedef_inherit.NewFoo()
	b := typedef_inherit.NewBar()

	x := typedef_inherit.Do_blah(a)
	if x != "Foo::blah" {
		panic(x)
	}

	x = typedef_inherit.Do_blah(b)
	if x != "Bar::blah" {
		panic(x)
	}

	c := typedef_inherit.NewSpam()
	d := typedef_inherit.NewGrok()

	x = typedef_inherit.Do_blah2(c)
	if x != "Spam::blah" {
		panic(x)
	}

	x = typedef_inherit.Do_blah2(d)
	if x != "Grok::blah" {
		panic(x)
	}

	x = d.Far()
	if x != "Spam::far" {
		panic(x)
	}
}
