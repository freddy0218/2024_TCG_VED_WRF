for comb_num in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 #0 
do 
for coeff in 0.9 0.65 0.55 0.45 0.35 0.3 0.25 0.95 0.85 0.8 0.75 0.7 0.6 0.5 0.4
	do
		sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/ml_scripts --job-name MLved_resumetrain --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/ml_scripts/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/ml_scripts/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 64G --time 3:00:00 --wrap "module purge; module load gcc; source ~/.bashrc ; conda activate /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/miniconda3/envs/fred_workenv ;python /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/ml_scripts/train_resumeVED.py $comb_num $coeff 6100"
done
done
