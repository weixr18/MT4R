# 运行 REINFORCE「去 c_term / 稳杆为主」重训的单个种子（REINFORCE-retrain-plan.md v2）。
# 用法: pwsh -File run_noc_term_seed.ps1 -Seed <k> -Lr <lr> -Iters <n> -MaxSteps <n> -Tag <smoke/lrX | formal>
# 产物写到 src-rl/res/s_rf_retrain_l3_noc_term/<Tag>/seed<k>/{log.txt,reward_history_reinforce_cartpole.{json,png},ckpt/}
param(
    [Parameter(Mandatory=$true)][int]$Seed,
    [Parameter(Mandatory=$true)][double]$Lr,
    [Parameter(Mandatory=$true)][int]$Iters,
    [Parameter(Mandatory=$true)][int]$MaxSteps,
    [Parameter(Mandatory=$true)][string]$Tag
)

$py = 'E:\Anaconda3\envs\py311-gym\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# 本仓库根 = 脚本所在目录（4-MT4R-github/CartPole/），与 cwd 无关
$repo = $PSScriptRoot
$base = Join-Path $repo 'src-rl\res\s_rf_retrain_l3_noc_term'
$runDir = Join-Path $base (Join-Path $Tag ("seed{0}" -f $Seed))
$ckptDir = Join-Path $runDir 'ckpt'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
$log = Join-Path $runDir 'log.txt'

$argsList = @(
    (Join-Path $repo 'src-rl\1_reinforce.py'),
    '--env','cartpole',
    '--iters',"$Iters",'--eps-per-iter','32','--lr',("$Lr"),
    '--init-log-std','0','--gamma','1','--max-steps',"$MaxSteps",
    '--reward-coefs','0.2,0.05,1.0,0.01',
    '--init-theta-range','-40,40','--init-x-range','-1.5,1.5',
    '--init-xdot-range','-1.0,1.0','--init-thetadot-range','-10,10',
    '--ckpt-interval','200',
    '--ckpt-dir',$ckptDir,
    '--save-dir',$runDir,
    '--seed',("$Seed")
)

& $py @argsList *>&1 | Out-File -FilePath $log -Encoding utf8
Write-Output ("done seed {0} lr {1} tag {2}; log={3}" -f $Seed, $Lr, $Tag, $log)
