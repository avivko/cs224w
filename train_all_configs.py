import os

PWD='/home/users/kormanav/cs224w'
CONDA_ENV = 'cs224wproj'
TORCH_HONE = PWD + '/torch_home'
PATH_REPO = PWD +'/cs224w'
PATH_SCRIPT = PATH_REPO + '/src/config_to_pyg_train.py'
CONFIGS = ["config_1A", "config_1B","config_1C",
           "config_2A","config_2B", "config_2C",
           "config_3A","config_3B","config_3C"]
SLURM_OUT = PWD + '/slurm_out'

def make_bash(py_script, config, job_name, slurm_out, torch_home, conda_env):

    j_name = job_name
    p_sbatch_script = f"{slurm_out}/{j_name}.sh"
    sbatch_file = open(p_sbatch_script, 'w')
    sbatch_file.write("#!/bin/bash \n")
    sbatch_file.write(f"#SBATCH --job-name={j_name} \n")
    sbatch_file.write("#SBATCH --gres=gpu:1 \n")
    sbatch_file.write("#SBATCH --cpus-per-task=16 \n")
    sbatch_file.write("#SBATCH --time=02:00:00 \n")
    sbatch_file.write("#SBATCH --partition=owners,gpu \n")
    sbatch_file.write("#SBATCH --mem-per-cpu=15GB \n")
    sbatch_file.write(f"#SBATCH --error={slurm_out}/{j_name}_job_%J.err \n")
    sbatch_file.write(f"#SBATCH --output={slurm_out}/{j_name}_job_%J.out \n")
    sbatch_file.write("\n")
    sbatch_file.write("module load cuda/11.3.1 \n")
    sbatch_file.write("\n")
    sbatch_file.write("source $HOME/.bashrc \n")
    sbatch_file.write(f"conda activate {conda_env} \n")
    sbatch_file.write("\n")
    sbatch_file.write(f"export TORCH_HOME={torch_home} \n")
    sbatch_file.write("\n")
    sbatch_file.write(
        f"python {py_script} {config}\n")
    sbatch_file.write("\n")
    sbatch_file.close()

    print(f"Submitting job {j_name}")
    os.system(f'sbatch {p_sbatch_script}')

if __name__ == '__main__':
    for cfg in CONFIGS:
        make_bash(PATH_SCRIPT,
                  cfg,
                  job_name=cfg,
                  slurm_out=SLURM_OUT,
                  torch_home=TORCH_HONE,
                  conda_env=CONDA_ENV
                  )