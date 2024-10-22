for comb_num in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 #6 7 8 9 #1 2 4 5 
	do
		sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts --job-name wrfproc_test --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 256G --time 4:00:00 --wrap "module purge; module load gcc; source ~/.bashrc ; conda activate /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/miniconda3/envs/fred_workenv ;python /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts/train_PCA.py $comb_num 3050 3 5"
done
