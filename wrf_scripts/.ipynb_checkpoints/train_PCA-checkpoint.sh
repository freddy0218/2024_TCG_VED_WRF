for varname in U V H_DIABATIC REFL_10CM RTHRATLW RTHRATSW
do	
	for memb in 10 #6 7 8 9 #1 2 4 5 
	do
		sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts --job-name wrfproc_test --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 64G --time 32:00:00 --wrap "module purge; module load gcc; source ~/.bashrc ; conda activate /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/miniconda3/envs/fred_workenv ;python /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/wrf_scripts/get_onevar.py $varname $memb"
	done
done
