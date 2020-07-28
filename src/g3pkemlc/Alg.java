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
import g3pkemlc.utils.Utils.KMode;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import net.sf.jclec.algorithm.classic.SGE;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.stringtree.StringTreeCreator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.util.IndividualStatistics;
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
	 * Betters selector.
	 */
	BettersSelector bselector = new BettersSelector(this);
	
	/**
	 * Max number of children at each node.
	 * By default, it is set to 7
	 */
	int maxChildren;
	
	/**
	 * Max depth of the tree.
	 * By default, it is set to 3.
	 */
	int maxDepth;
	
	/**
	 * Minimum size of the k-labelsets.
	 * If not set, it will be 3.
	 */
	int minK;
	
	/**
	 * Maximum size of the k-labelsets.
	 * If not set, it will be numLabels/2.
	 */
	int maxK;
	
	/**
	 * Mode of k selection for each k-labelset.
	 * By default it is gaussian.
	 */
	KMode kMode;
	
	/**
	 * Expected average number of votes per label in the pool
	 * By default, set to 20.
	 */
	int v;
	
	/**
	 * Full training dataset.
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Current sampled training data for a given multi-label classifier.
	 */
	MultiLabelInstances currentTrainData;
	
	/**
	 * Ratio of instances sampled at each train data.
	 * By default, it is 0.75 (75% of instances for each classifier).
	 */
	float sampleRatio;
	
	/**
	 * Test dataset.
	 */
	MultiLabelInstances testData;
	
	/**
	 * Number of different MLC in the pool.
	 * It varies depending on the expected average number of votes in the initial pool.
	 */
	int nMLC;
	
	/**
	 * Utils.
	 */
	Utils utils;

	/**
	 * Table with predictions of each classifier.
	 */
	Hashtable<String, Prediction> tablePredictions;
	
	/**
	 * Random numbers generator.
	 */
	IRandGen randgen;
	
	/**
	 * Seed for random numbers.
	 */
	int seed;
	
	/**
	 * Use confidences or bipartitions in combining predictions.
	 * By default it is false, so each classifier as well as combination nodes
	 * 	consider bipartitions instead of confidences.
	 */
	boolean useConfidences;
	
	/**
	 * Array of k-labelsets built in the pool.
	 */
	ArrayList<KLabelset> klabelsets;
	
	/**
	 * Final ensemble.
	 */
	EMLC ensemble;
	
	/**
	 * Lock for parallel critical code.
	 */
	Lock lock = new ReentrantLock();
	
	/**
	 * Beta for fitness.
	 * By default it is 0.5, i.e., same weight to each metric in fitness.
	 */
	float beta;
	
	/**
	 * Stop condition: Number of iterations without improvement of the best.
	 * By default it is set to 10.
	 */
	int nItersWithoutImprovement = 10;
	
	/**
	 * Best fitness so far.
	 */
	float bestFitness = -1;
	
	/**
	 * Iteration in which best fitness was achieved.
	 */
	int lastIterBestFitness = 0;
	
	/**
	 * Best average fitness value. Used to modify operators probabilities.
	 */
	float bestAvgFitness = 0;
	
	/**
	 * Indicate if individuals are created biased by phi-value of labels.
	 * By default, it is true.
	 */
	boolean phiBasedPool = true;
	
	/**
	 * Getter for test data.
	 * 
	 * @return Test data
	 */
	public MultiLabelInstances getTestData() {
		return testData;
	}
	
	/**
	 * Getter for the seed.
	 * 
	 * @return Seed
 	 */
	public int getSeed() {
		return seed;
	}
	
	/**
	 * Getter for the ensemble.
	 * 
	 * @return Ensemble
	 */
	public EMLC getEnsemble() {
		return ensemble;
	}
	
	/**
	 * Configure some default aspects and parameters of EME to make the configuration easier.
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
		if(! configuration.containsKey("recombinator[@rec-prob]")) {
			configuration.addProperty("recombinator[@rec-prob]", "0.5");
		}
		
		if(! configuration.containsKey("mutator[@type]")) {
			configuration.addProperty("mutator[@type]", "g3pkemlc.mutator.Mutator");
		}
		if(! configuration.containsKey("mutator[@mut-prob]")) {
			configuration.addProperty("mutator[@mut-prob]", "0.5");
		}
		
		//Use confidences (only if not provided)
		if(! configuration.containsKey("use-confidences")) {
			configuration.addProperty("use-confidences", "false");
		}
		
		//Listener type (only if not provided)
		if(! configuration.containsKey("listener[@type]")) {
			configuration.addProperty("listener[@type]", "g3pkemlc.Listener");
		}
		
		//Average votes per label in initial pool
		if(! configuration.containsKey("v")) {
			configuration.addProperty("v", "20");
		}
		
		//k
		if(! configuration.containsKey("min-k")) {
			configuration.addProperty("min-k", "3");
		}
		if(! configuration.containsKey("max-k")) {
			configuration.addProperty("max-k", "-1");
		}
		
		//k-mode for selecting size of each k-labelset
		if(! configuration.containsKey("k-mode")) {
			configuration.addProperty("k-mode", "gaussian");
		}
		
		//Max children and max depth for tree
		if(! configuration.containsKey("max-depth")) {
			configuration.addProperty("max-depth", "3");
		}
		if(! configuration.containsKey("max-children")) {
			configuration.addProperty("max-children", "7");
		}
		
		//Sampling ratio
		if(! configuration.containsKey("sampling-ratio")) {
			configuration.addProperty("sampling-ratio", "0.75");
		}
		
		//Phi-based initial k-labelsets
		if(! configuration.containsKey("phi-based-klabelsets")) {
			configuration.addProperty("phi-based-klabelsets", "true");
		}
		
		//Beta for fitness
		if(! configuration.containsKey("beta")) {
			configuration.addProperty("beta", "0.5");
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
		
		
		sampleRatio = configuration.getFloat("sampling-ratio");
		if(sampleRatio <= 0 || sampleRatio > 1) {
			System.out.println("Sample ratio must be a value in the (0, 1] range.");
		}
		
		minK = configuration.getInt("min-k");
		maxK = configuration.getInt("max-k");
		
		maxChildren = configuration.getInt("max-children");
		maxDepth = configuration.getInt("max-depth");
		useConfidences = configuration.getBoolean("use-confidences");
		beta = configuration.getFloat("beta");
		
		phiBasedPool = configuration.getBoolean("phi-based-klabelsets");
		
		String kModeString = configuration.getString("k-mode");
		if(kModeString.equalsIgnoreCase("uniform")) {
			kMode = KMode.uniform;
		}
		else if(kModeString.equalsIgnoreCase("gaussian")) {
			kMode = KMode.gaussian;
		}
		else {
			System.out.println(kModeString + " is not a valid value for k-mode.");
			System.exit(-1);
		}
		
		//Get datasets
		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		fullTrainData = null;
		currentTrainData = null;
		testData = null;
		try {
			fullTrainData = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			testData = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);
			
			int nLabels = fullTrainData.getNumLabels();
			if(maxK < 0) {
				maxK = (int)Math.floor(nLabels*0.5);
			}
			
			if(minK < 2 || minK > maxK || minK > nLabels) {
				System.out.println("Incorrect value for minK.");
			}
			if(maxK < minK || maxK > nLabels) {
				System.out.println("Incorrect value for maxK.");
			}
			
			v = configuration.getInt("v");
			if(v < 1) {
				System.out.println("Incorrect value for v.");
			}
			
			//Create folder for classifiers if it does not exist
			File f = new File("mlc/");
			if (!f.exists()) {
			   f.mkdir();
			}
			
			//Generate k-labelsets
			KLabelsetGenerator klabelsetGen = new KLabelsetGenerator(minK, maxK, nLabels, kMode);
			klabelsetGen.setRandgen(randgen);
			if(phiBasedPool) {
				//Get phi matrix
				Statistics stat = new Statistics();
				double [][] phi = stat.calculatePhi(fullTrainData);
				
				//Change NaNs by 0
				for(int i=0; i<phi.length; i++) {
					for(int j=0; j<phi[0].length; j++) {
						if(Double.isNaN(phi[i][j])) {
							phi[i][j] = 0.0;
						}
					}
				}
				
				klabelsetGen.setPhiBiased(true, phi);
			}
			else {
				klabelsetGen.setPhiBiased(false, null);
			}
			
			//Generate k-labelsets
			klabelsets = klabelsetGen.generateKLabelsets(v);
			
			//Get number of classifiers finally created
			nMLC = klabelsets.size();

			//Print the k-labelsets
			System.out.println("nMLC: " + nMLC);
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
		
		((Mutator)mutator.getDecorated()).setnChildren(maxChildren);
		((Mutator)mutator.getDecorated()).setMaxTreeDepth(maxDepth);
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
		
		//Get current best fitness with 4 decimal points
		float currentBestFitness = (float)((SimpleValueFitness)((StringTreeIndividual)bselector.select(bset, 1).get(0)).getFitness()).getValue();
		currentBestFitness *= 10000;
		currentBestFitness = Math.round(currentBestFitness);
		currentBestFitness /= 10000;
		
		//Check if it has improved the best so far
		if(currentBestFitness > bestFitness) {
			bestFitness = currentBestFitness;
			lastIterBestFitness = generation;
		}
		
		//Modify crossover/mutation probabilities
		float step = (float)0.02;
		float avgFitness = (float)IndividualStatistics.averageFitnessAndFitnessVariance(bset)[0];
		//If the average fitness is better than the best average so far, increase crossover probability
		if(avgFitness > bestAvgFitness) {
			bestAvgFitness = avgFitness;
			if(getRecombinationProb() < 1-(step*2) && getMutationProb() >= step*2) {
				setRecombinationProb(getRecombinationProb() + 0.02);
				setMutationProb(getMutationProb() - 0.02);
			}
		}
		else { //If avg fitness is not best so far, increase mutation
			if(getMutationProb() < 1-(step*2) && getRecombinationProb() >= step*2) {
				setRecombinationProb(getRecombinationProb() - 0.02);
				setMutationProb(getMutationProb() + 0.02);
			}
		}
	
		//Stop condition
		if ((generation >= (lastIterBestFitness+nItersWithoutImprovement) && bestFitness > 0)|| generation >= maxOfGenerations) {			
			System.out.println("Finished in generation " + generation);
			
			//Get base learner
			MultiLabelLearner learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(seed);
			
			//Generate ensemble object
			ensemble = new EMLC(learner, klabelsets, bestGenotype, useConfidences);
			
			System.out.println("Votes per label: " + Arrays.toString(TreeUtils.votesPerLabel(bestGenotype, klabelsets, fullTrainData.getNumLabels())));
			
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
	
//	@Override
//	protected void doUpdate() {
//		int nElite = (int)Math.round(populationSize*.1);
//		if(nElite < 1) {
//			nElite = 1;
//		}
//		List<IIndividual> bestb = bselector.select(bset, nElite);
//		List<IIndividual> bestc = bselector.select(cset, populationSize-nElite);
//		
//		bset.clear();
//		bset.addAll(bestb);
//		bset.addAll(bestc);
//		
//		// Clear pset, rset & cset
//		pset = null;
//		rset = null;
//		cset = null;	
//	}
	
	/**
	 * Clear some variables to null
	 */
	public void clear() {
		fullTrainData = null;
		currentTrainData = null;
		tablePredictions = null;
		klabelsets = null;
		testData = null;
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
			float[][] currentPredictions = new float[currentFullData.getNumInstances()][klabelsets.get(c).k];
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
