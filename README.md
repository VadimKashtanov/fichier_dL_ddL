Ce scipte montre que mon algorithme de la Hessienne "retro-retro-propagation" est exacte.

Il calcule la hessienne de f(ax+b) ou a,x,b sont des matrices.

Il compare la Hessienne aux calcules infinitésimaux :
* Difference de gradient, donc un vecteur de vecteur gradient d(grad(xi)) / dxj
* Ou directement de maniere 100% infinitésimale : (f(xi+1e-3, xj+1e-3)-f(xi+1e-3,xj) - f(xi,xj+1e-3) - f(xi,xj)) / (1e-3 * 1e-3)  ou 1e-3 est le petit h -> 0
