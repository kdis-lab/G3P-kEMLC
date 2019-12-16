package g3pkemlc;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.commons.configuration.Configuration;

import g3pkemlc.mutator.Mutator;
import g3pkemlc.recombinator.Crossover;
import g3pkemlc.utils.DatasetTransformation;
import g3pkemlc.utils.KLabelset;
import g3pkemlc.utils.KLabelsetGenerator;
import g3pkemlc.utils.MulanUtils;
import g3pkemlc.utils.TreeUtils;
import g3pkemlc.utils.Utils;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.algorithm.classic.SGE;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.stringtree.StringTreeCreator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;

/**
 * Class implementing the evolutionary algorithm.
 * 
 * @author Jose M. Moyano
 *
 */
public class Alg extends SGE {

	/**
	 * serialVersionUID
	 */
	private static final long serialVersionUID = -790335501425435317L;

	/**
	 * Betters selector
	 */
	BettersSelector bselector = new BettersSelector(this);
	
	/**
	 * Max number of children at each node
	 */
	int maxChildren;
	
	/**
	 * Max depth of the tree
	 */
	int maxDepth;
	
	/**
	 * Minimum size of the k-labelsets
	 */
	int minK;
	
	/**
	 * Maximum size of the k-labelsets
	 */
	int maxK;
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Current sampled training data for a given multi-label classifier
	 */
	MultiLabelInstances currentTrainData;
	
	/**
	 * Ratio of instances sampled at each train data
	 */
	double sampleRatio;
	
	/**
	 * Test dataset
	 */
	MultiLabelInstances testData;
	
	/**
	 * Number of different MLC
	 */
	int nMLC;
	
	/**
	 * Utils
	 */
	Utils utils;

	/**
	 * Table with predictions of each classifier
	 */
	Hashtable<String, Prediction> tablePredictions;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Multi-label base classifier
	 */
	MultiLabelLearnerBase learner;
	
	/**
	 * Seed for random numbers
	 */
	int seed;
	
	/**
	 * Use confidences or bipartitions in combining predictions
	 */
	boolean useConfidences;
	
	/**
	 * Array of k-labelsets
	 */
	ArrayList<KLabelset> klabelsets;
	
	/**
	 * Final ensemble
	 */
	EMLC ensemble;
	
	/**
	 * Lock for parallel critical code
	 */
	Lock lock = new ReentrantLock();
	
	/**
	 * Beta for fitness
	 */
	double beta;
	
	/**
	 * Getter for test data
	 * 
	 * @return Test data
	 */
	public MultiLabelInstances getTestData() {
		return testData;
	}
	
	/**
	 * Getter for the seed
	 * 
	 * @return Seed
 	 */
	public int getSeed() {
		return seed;
	}
	
	/**
	 * Getter for the ensemble
	 * 
	 * @return Ensemble
	 */
	public EMLC getEnsemble() {
		return ensemble;
	}
	
	/**
	 * Configure some default aspects and parameters of EME to make the configuration easier
	 * 
	 * @param configuration Configuration
	 */
	private void configureDefaults(Configuration configuration) {
		//Species
		configuration.setProperty("species[@type]", "net.sf.jclec.stringtree.StringTreeIndividualSpecies");
		configuration.setProperty("species[@genotype-length]", "0");
		
		//Evaluator (only if not provided)
		if(! configuration.containsKey("evaluator[@type]")) {
			configuration.addProperty("evaluator[@type]", "g3pkemlc.Evaluator");
		}
		
		//Provider (only if not provided)
		if(! configuration.containsKey("provider[@type]")) {
			configuration.addProperty("provider[@type]", "net.sf.jclec.stringtree.StringTreeCreator");
		}
		
		//Randgen type (only if not provided)
		if(! configuration.containsKey("rand-gen-factory[@type]")) {
			configuration.addProperty("rand-gen-factory[@type]", "g3pkemlc.RanecuFactory2");
		}
		
		//Parents-selector (only if not provided)
		if(! configuration.containsKey("parents-selector[@type]")) {
			configuration.addProperty("parents-selector[@type]", "net.sf.jclec.selector.TournamentSelector");
		}
		if(! configuration.containsKey("parents-selector.tournament-size")) {
			configuration.addProperty("parents-selector.tournament-size", "2");
		}
		
		//Crossover and mutation operators (only if not provided)
		if(! configuration.containsKey("recombinator[@type]")) {
			configuration.addProperty("recombinator[@type]", "g3pkemlc.recombinator.Crossover");
		}
		if(! configuration.containsKey("mutator[@type]")) {
			configuration.addProperty("mutator[@type]", "g3pkemlc.mutator.Mutator");
		}
		
		//Use confidences (only if not provided)
		if(! configuration.containsKey("use-confidences")) {
			configuration.addProperty("use-confidences", "false");
		}
		
		//Listener type (only if not provided)
		if(! configuration.containsKey("listener[@type]")) {
			configuration.addProperty("listener[@type]", "g3pkemlc.Listener");
		}
	}
	
	@Override
	public void configure(Configuration configuration) {
		configureDefaults(configuration);
		super.configure(configuration);
		
		seed = configuration.getInt("rand-gen-factory[@seed]");
		randgen = randGenFactory.createRandGen();
		utils = new Utils(randgen);
		
		
		//Initialize table for predictions
		tablePredictions = new Hashtable<String, Prediction>();
		
		//Get datasets
		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
		minK = configuration.getInt("min-k");
		maxK = configuration.getInt("max-k");
		nMLC = configuration.getInt("different-classifiers");
		maxChildren = configuration.getInt("max-children");
		maxDepth = configuration.getInt("max-depth");
		useConfidences = configuration.getBoolean("use-confidences");
		beta = configuration.getDouble("beta");
		
		fullTrainData = null;
		currentTrainData = null;
		testData = null;
		try {
			fullTrainData = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			testData = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);

			//Create folder for classifiers if it does not exist
			File f = new File("mlc/");
			if (!f.exists()) {
			   f.mkdir();
			}
			
			//Generate k-labelsets
			KLabelsetGenerator klabelsetGen = new KLabelsetGenerator(minK, maxK, fullTrainData.getNumLabels(), nMLC);
			klabelsetGen.setRandgen(randgen);
			klabelsets = klabelsetGen.generateKLabelsets();
			klabelsetGen.printKLabelsets();
			
			//Set number of threads
			ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
			
			//Create, store, and get predictions of each different classifier
			for(int c=0; c<nMLC; c++) {
				executorService.execute(new BuildClassifierParallel(c));				
			}
			executorService.shutdown();
			
			try {
				//Wait until all threads finish
				executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			currentTrainData = null;
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//Set settings of provider, genetic operators and evaluator
		((StringTreeCreator)provider).setMaxChildren(maxChildren);
		((StringTreeCreator)provider).setMaxDepth(maxDepth);
		((StringTreeCreator)provider).setnMax(nMLC);
		
		((Mutator)mutator.getDecorated()).setMaxTreeDepth(maxDepth);
		((Mutator)mutator.getDecorated()).setnChilds(maxChildren);
		((Mutator)mutator.getDecorated()).setnMax(nMLC);
		
		((Crossover)recombinator.getDecorated()).setMaxTreeDepth(maxDepth);
		
		((Evaluator)evaluator).setFullTrainData(fullTrainData);
		((Evaluator)evaluator).setTablePredictions(tablePredictions);
		((Evaluator)evaluator).setUseConfidences(useConfidences);
		((Evaluator)evaluator).setBeta(beta);
		((Evaluator)evaluator).setRandgen(randgen);
	}
	
	@Override
	protected void doInit() {
		super.doInit();
	}
	
	@Override
	protected void doControl()
	{		
		//Get genotype of best individual
		String bestGenotype = ((StringTreeIndividual)bselector.select(bset, 1).get(0)).getGenotype();
		
		if (generation >= maxOfGenerations) {			
			//Get base learner
			learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(seed);
			
			//Generate ensemble object
			ensemble = new EMLC(learner, klabelsets, bestGenotype, useConfidences);
			
			System.out.println(Arrays.toString(TreeUtils.votesPerLabel(bestGenotype, klabelsets, fullTrainData.getNumLabels())));
			
			//Print the leaves; i.e., different classifiers used in the ensemble
			System.out.println(utils.getLeaves(bestGenotype));
			
			try {
				//Build the ensemble
				ensemble.build(fullTrainData);
				
				//After building the ensemble, we can remove all the classifiers built and stored in hard disk
				utils.purgeDirectory(new File("mlc/"));				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			state = FINISHED;
			return;
		}
	}	
	
	/**
	 * Clear some variables to null
	 */
	public void clear() {
		fullTrainData = null;
		currentTrainData = null;
		tablePredictions = null;
		klabelsets = null;
		testData = null;
		learner = null;
		ensemble = null;
		System.gc();
	}
	
	/**
	 * Class to parallelize building base classifiers
	 * 
	 * @author Jose M. Moyano
	 *
	 */
	public class BuildClassifierParallel extends Thread {
		
		//Index of classifier to build
		int index;
		
		public BuildClassifierParallel(int index) {
			this.index = index;
		}
		
		public void run() {
			try {
				buildClassifier(index);
			}catch(Exception e) {
				e.printStackTrace();
				System.exit(-1);
			}
		}
	}
	
	/**
	 * Build the c-th classifier
	 * 
	 * @param c Index of classifier to build
	 */
	public void buildClassifier(int c) {
		IRandGen randgen = null;
		MultiLabelInstances currentTrainData, currentFullData;
		MultiLabelLearnerBase learner;
		int seed = c;
		
		try {
			randgen = new RanecuFactory2().createRandGen(seed, seed*2);
			
			//Sample c-th data
			currentTrainData = MulanUtils.sampleData(fullTrainData, sampleRatio, randgen);
			
			//Build classifier with c-th data
			learner = null;
			learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(seed);
			//Transform full train data
			DatasetTransformation dt = new DatasetTransformation();
			currentTrainData = dt.transformDataset(currentTrainData, klabelsets.get(c).getKlabelset());
			//Build
			learner.build(currentTrainData);
			
			//If data was sampled to build, transform fullTrain too; otherwise just use same data to gather predictions
			if(sampleRatio >= 0.999) {
				currentFullData = currentTrainData;
			}
			else {
				currentFullData = dt.transformDataset(fullTrainData, klabelsets.get(c).getKlabelset());
			}
			
			//Store object of classifier in the hard disk				
			utils.writeObject(learner, "mlc/classifier"+c+".mlc");
			
			//Get predictions of c-th classifier over all data
			double[][] currentPredictions = new double[currentFullData.getNumInstances()][klabelsets.get(c).k];
			for(int i=0; i<currentFullData.getNumInstances(); i++) {
				if(useConfidences) {
					System.arraycopy(learner.makePrediction(currentFullData.getDataSet().get(i)).getConfidences(), 0, currentPredictions[i], 0, currentFullData.getNumLabels());
				}
				else {
					System.arraycopy(utils.bipartitionToConfidence(learner.makePrediction(currentFullData.getDataSet().get(i)).getBipartition()), 0, currentPredictions[i], 0, currentFullData.getNumLabels());
				}
			}
			
			//Create Prediction object
			Prediction pred = new Prediction(dt.getOriginalLabelIndices(), currentPredictions);
			
			//Put predictions in table
			lock.lock();
			tablePredictions.put(String.valueOf(c), new Prediction(pred));
			lock.unlock();
			
			//Clear objects
			currentTrainData = null;
			currentFullData = null;
			pred = null;
			learner = null;
			currentPredictions = null;
			dt = null;
			System.gc();
			
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}		
	}	
}
