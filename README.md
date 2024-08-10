Para gerar instâncias aleatórias, rodar:

```bash 
$ python utils/gen_random_instance.py (output file) (k) (cov matrix) (point centers) --show
```

Por exemplo, rodando

```bash
$ python utils/gen_random_instance.py test.txt 3 3 "1,0,0,1" "(0,0)" "(1,1)" "(2,2)" -s  
```

Geramos uma instância com a matriz de covariância `[[1,0], [0,1]]` e `3` centros, com coordenadas em `(0,0)`, `(1,1)` e `(2,2)` respectivamente.