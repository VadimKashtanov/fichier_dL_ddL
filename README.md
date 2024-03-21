Soit une fonction F(x) avec la fonction de score S(x)

La technique principale pour calculer le gradient de x est la Retro-Propagation :

propagation :
```
y = F(x)
s = S(y)
```
retro-propagation :
```
dy = S'(x)
dx = dy * F'(x)
```
Nous auront donc un gradient vecteur dS/dx

Pour la Matrice Hessienne il faut : d( dS/dx[i] ) / dx[j] = dSdS/dx[i]dx[j] = d( d S(F(x[i]))  /dx[i] ) / dx[j] = d ( S'(F(x[i])) * F'(x[i]) ) / dx[j] = d(S')/dxj * F'  + S' * d(F')/dxj

Donc on retro-propage la retro-propagation :

```
y = F(x)
s = S(y)
dy = S'(x)
dx = dy * F'(x)



```


