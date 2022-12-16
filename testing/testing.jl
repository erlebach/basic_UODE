using Revise

using GordonPackage
import GordonPackage as gp
using Random
using Plots

greet()
GordonPackage.f()
gp.f()
gp.g()

function subtypetree(t, level=1, indent=4)
    level == 1 && println(t)
    for s in subtypes(t)
      println(join(fill(" ", level * indent)) * string(s))
      subtypetree(s, level+1, indent)
    end
end