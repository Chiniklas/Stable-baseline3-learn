import cpp11_rvalue_reference

a = cpp11_rvalue_reference.A()

a.setAcopy(5)
if a.getAcopy() != 5:
    raise RuntimeError("int A::getAcopy() value is ",
                         a.getAcopy(), " should be 5")

ptr = a.getAptr()

a.setAptr(ptr)
if a.getAcopy() != 5:
    raise RuntimeError("after A::setAptr(): int A::getAcopy() value is ", a.getAcopy(
    ), " should be 5")

a.setAref(ptr)
if a.getAcopy() != 5:
    raise RuntimeError("after A::setAref(): int A::getAcopy() value is ", a.getAcopy(
    ), " should be 5")

rvalueref = a.getAmove()

a.setAref(rvalueref)
if a.getAcopy() != 5:
    raise RuntimeError("after A::setAmove(): int A::getAcopy() value is ", a.getAcopy(
    ), " should be 5")
