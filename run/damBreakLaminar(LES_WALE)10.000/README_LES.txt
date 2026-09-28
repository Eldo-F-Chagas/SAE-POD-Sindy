damBreak LES-WALE — OpenFOAM 13
================================

Este pacote converte a configuração do toy case damBreak para LES usando
o modelo submalha WALE, mantendo o solver incompressibleVoF.

ARQUIVOS ALTERADOS/CRIADOS
--------------------------
0/nut
constant/momentumTransport
system/controlDict
system/fvSchemes
system/fvSolution

ARQUIVOS MANTIDOS DO SEU CASO
-----------------------------
system/blockMeshDict
system/setFieldsDict

IMPORTANTE SOBRE A PASTA 0
--------------------------
Mantenha os arquivos originais:
    0/U
    0/p_rgh
    0/alpha.water

Adicione:
    0/nut

REMOVA, SE AINDA ESTIVEREM DA SIMULACAO URANS:
    0/k
    0/omega

O modelo WALE nao resolve equacoes de transporte para k e omega.

EXECUCAO
--------
blockMesh
checkMesh
setFields
foamRun -solver incompressibleVoF | tee log.incompressibleVoF

Ou:
chmod +x Allrun_LES
./Allrun_LES

CONFIGURACAO LES
----------------
Modelo SGS: WALE
delta: cubeRootVol
Ck: 0.094
Cw: 0.325

Tempo:
CrankNicolson 0.9
maxCo = 0.5
maxAlphaCo = 0.5
maxDeltaT = 1e-4 s

Conveccao de velocidade:
Gauss LUST grad(U)

ATENCAO CIENTIFICA
------------------
A malha fornecida continua sendo 2D (1 celula na direcao z e faces empty).
Esta configuracao e util para verificar que o caso LES esta montado e roda,
mas NAO deve ser tratada como uma LES fisicamente validada.

Uma LES fisicamente defensavel precisa ser 3D e ter resolucao espacial
suficiente para resolver as grandes escalas turbulentas. Depois do teste
de execucao, o proximo passo deve ser definir a profundidade 3D e o numero
de celulas na direcao z, seguido por estudo de malha, Courant, nut/nu,
espectro de energia e comparacao com benchmark do dam-break.
