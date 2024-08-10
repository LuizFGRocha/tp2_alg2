Para gerar instâncias aleatórias, rodar:

```bash 
$ python utils/gen_random_instance.py (output file) (k) (points per center) (cov matrix) (point centers) --show
```

Por exemplo, rodando

```bash
$ python utils/gen_random_instance.py test.txt 3 4 "1,0,0,1" "(0,0)" "(1,1)" "(2,2)" -s  
```

Geramos uma instância com a matriz de covariância `[[1,0], [0,1]]` e `3` centros, cada um com `4` pontos, com coordenadas de origem em `(0,0)`, `(1,1)` e `(2,2)` (média na distribuição normal multivariada) respectivamente.