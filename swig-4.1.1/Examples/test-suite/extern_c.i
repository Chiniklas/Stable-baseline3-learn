%module extern_c

%inline %{
extern "C" {
void RealFunction(int value);
typedef void Function1(int value); // Fails
typedef int Integer1;
int Integer3;
}
typedef void Function2(int value); // Works
typedef int Integer2;
%}

%{
void RealFunction(int value) {}
%}


%inline %{
extern "C" {
  typedef void (*Hook1_t)(int, const char *);
}
extern "C" typedef void (*Hook2_t)(int, const char *);
void funcy1(Hook1_t) {}
void funcy2(Hook2_t) {}
Hook1_t hook1;
Hook2_t hook2;

extern "C" typedef int Integer;
Integer int1;
extern "C" int int2;
extern "C" { extern int int3; }
extern "C" { int int4 = 789; }
%}

%{
extern "C" {
  int int2 = 123;
  int int3 = 456;
}
%}
