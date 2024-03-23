Ce scipte montre que mon algorithme de la Hessienne "retro-retro-propagation" est exacte.

Il calcule la hessienne de f(ax+b) ou a,x,b sont des matrices.

Il compare la Hessienne aux calcules infinitésimaux :
* Difference de gradient, donc un vecteur de vecteur gradient d(grad(xi)) / dxj
* Ou directement de maniere 100% infinitésimale : (f(xi+1e-3, xj+1e-3)-f(xi+1e-3,xj) - f(xi,xj+1e-3) - f(xi,xj)) / (1e-3 * 1e-3)  ou 1e-3 est le petit h -> 0


Resultat de ce petit exemple : 
```
 === Start === 
====================================
2.5471205711364746
====================================
0.06205892562866211
k =  41.043581488624405
```
2.54 est le temps avec une methode d'itération de retro-propagation

0.06 est le temps avec ma méthode de retro-retro-propagation

Ici il n'est que 41 fois plus rapide, mais plus la fonction F est grande plus le `k` sera grand.
