cd ../
install_dir=$(pwd)
cd watkit_analysis_container/
date=$(date "+%d_%m_%y_%H_%M")
echo ""
echo "Installation directory is $install_dir"
echo ""
echo "Backing Up CHARMM Components..."
backupdir=backup/$date
mkdir $backupdir
echo "Backupdir is  $backupdir"
echo ""
cp $install_dir/build/gnu/objlibs.mk $backupdir/
cp $install_dir/build/gnu/pref.dat $backupdir/
cp $install_dir/install.com $backupdir/
cp $install_dir/source/charmm/charmm_main.src $backupdir/
echo "Modifying Source..."
echo ""
echo "Installing SA_Analysis Features..."
echo ""
rm $install_dir/build/gnu/pref.dat
cp file_cabinet/wat_charmm_install_files/wat_objlibs.mk $install_dir/build/gnu/objlibs.mk
cp file_cabinet/wat_charmm_install_files/wat_install.com $install_dir/install.com
cp file_cabinet/wat_charmm_install_files/wat_charmm_main.src $install_dir/source/charmm/charmm_main.src
cp file_cabinet/wat_charmm_install_files/wat_analysis_ltm.src $install_dir/source/ltm/
cp file_cabinet/make.sh $install_dir/wat_analysis_make.sh
cp -r file_cabinet/toolkit $install_dir/source/
cp -r file_cabinet/md_conversion/charmm_conversion $install_dir/source/
cp file_cabinet/wat_make.sh $install_dir/wat_make.sh
echo "Compiling CHARMM..."
echo ""
cd ../
./wat_make.sh
cd watkit_analysis_container/
