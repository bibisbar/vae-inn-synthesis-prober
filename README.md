# vae-inn-synthesis-prober

# setup
- activate conda env

        conda env create -f vaeinn.yaml
        conda activate vae-inn
  
- clone INN repo and install reqs

        git clone https://github.com/ELIFE-ASU/INNLab
        cd INNLab
        python setup.py install
        cd ..
  
# commands
- show logs in tensorboards

        tensorboard --logdir ./tb_logs

