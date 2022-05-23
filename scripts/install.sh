# source scrips/install.sh
PROJECT=deeprec-prl22

# base directory where the project is
[[ ! -z "$1" ]] && BASEDIR=$1 || BASEDIR=$HOME
[[ ! -z "$2" ]] && PROJECTDIR=$2 || PROJECTDIR=$BASEDIR/$PROJECT
ENVDIR=$BASEDIR/envs/$PROJECT # directory for the virtual environemnt
QSOPTDIR=$BASEDIR/qsopt
CONCORDEDIR=$BASEDIR/concorde
PYTHON_VERSION=6
ORANGE='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

echo -e "${ORANGE}1) Preparing environment${NC}"
mkdir -p $ENVDIR
#sudo add-apt-repository ppa:deadsnakes/ppa
#sudo apt-get update
sudo apt install python3.$PYTHON_VERSION python3.$PYTHON_VERSION-dev python3.$PYTHON_VERSION-tk python3.$PYTHON_VERSION-venv curl python3-pip -y 
# sudo apt install mscorefonts
# sudo pip3 install -U python3-virtualenv
# sudo apt install python3-virtualenv
#virtualenv --system-site-packages -p python3.$PYTHON_VERSION $BASEDIR/envs/$PROJECT
python3.$PYTHON_VERSION -m venv $ENVDIR

echo -e "${ORANGE}2) Installing Concorde${NC}"

echo -e "${BLUE}=> download${NC}"
mkdir -p $QSOPTDIR
curl -o $QSOPTDIR/qsopt.a http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a
curl -o $QSOPTDIR/qsopt.h http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h
curl -o $BASEDIR/concorde.tgz http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar -xf $BASEDIR/concorde.tgz -C $BASEDIR
rm -rf $BASEDIR/concorde.tgz

echo -e "${BLUE}=> configuration${NC}"
cd $CONCORDEDIR
./configure --with-qsopt=$QSOPTDIR

echo -e "${BLUE}=> compilation${NC}"
make

echo -e "${BLUE}=> adjusting PATH${NC}"
if ! grep -q "$CONCORDEDIR/TSP" $ENVDIR/bin/activate ; then
   echo export PATH=\$PATH:$CONCORDEDIR/TSP >> $ENVDIR/bin/activate
fi

#if ! grep -q $PROJECTDIR $ENVDIR/bin/activate ; then
#   echo export PYTHONPATH=$PROJECTDIR >> $ENVDIR/bin/activate
#fi

echo -e "${ORANGE} 3) Installing Python requirements${NC}"
# sudo apt install enchant -y
# sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev -y
. $ENVDIR/bin/activate
cd $PROJECTDIR
pip install --upgrade pip
pip install -r requirements.txt
# python -m scripts.install_nltk
