package gpemlc;

import java.io.File;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.configuration.Configuration;

import gpemlc.Utils.ClassifierType;
import gpemlc.mutator.Mutator;
import gpemlc.recombinator.Crossover;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.classifier.transformation.PrunedSets;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.*;
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
	 * Type of classifier used: LP, CC, k-labelset
	 */
	Utils.ClassifierType classifierType;
	
	@Override
	public void configure(Configuration configuration) {
		super.configure(configuration);
		
		randgen = randGenFactory.createRandGen();
		utils = new Utils(randgen);
		seed = configuration.getInt("rand-gen-factory[@seed]");
		
		//Initialize table for predictions
		tablePredictions = new Hashtable<String, Prediction>();
		
		//Get datasets
		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
		nMLC = configuration.getInt("different-classifiers");
		maxChildren = configuration.getInt("max-children");
		maxDepth = configuration.getInt("max-depth");
		useConfidences = configuration.getBoolean("use-confidences");
		
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
			
			//Create, store, and get predictions of each different classifier
			for(int c=0; c<nMLC; c++) {
				//Sample c-th data
				currentTrainData = MulanUtils.sampleData(fullTrainData, sampleRatio, randgen);
				
				//Build classifier with c-th data
				learner = null;
				String learnerType = configuration.getString("base-learner");
				switch (learnerType.toUpperCase()) {
				case "BR":
					classifierType = ClassifierType.BR;
					learner = new BinaryRelevance(new J48());
					break;
				
				case "LP":
					classifierType = ClassifierType.LP;
					learner = new LabelPowerset2(new J48());
					((LabelPowerset2)learner).setSeed(seed);
					break;
				
				case "CC":
					classifierType = ClassifierType.CC;
					learner = new ClassifierChain(new J48(), utils.randomPermutation(fullTrainData.getNumLabels(), randgen));
					break;
					
				case "PS":
					classifierType = ClassifierType.PS;
					learner = new PrunedSets();
					break;
					
				case "KLABELSET":
				case "K-LABELSET":
					classifierType = ClassifierType.kLabelset;
					learner = null;
					break;

				default:
					classifierType = null;
					learner = null;
					break;
				}
				
				learner.build(currentTrainData);
				
				//Store object of classifier in the hard disk				
				utils.writeObject(learner, "mlc/classifier"+c+".mlc");
				
				//Get predictions of c-th classifier over all data
				Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
				for(int i=0; i<fullTrainData.getNumInstances(); i++) {
					if(useConfidences) {
						System.arraycopy(learner.makePrediction(fullTrainData.getDataSet().get(i)).getConfidences(), 0, pred.pred[i], 0, fullTrainData.getNumLabels());
					}
					else {
						System.arraycopy(utils.bipartitionToConfidence(learner.makePrediction(fullTrainData.getDataSet().get(i)).getBipartition()), 0, pred.pred[i], 0, fullTrainData.getNumLabels());
					}
				}
				
				//Put predictions in table
				tablePredictions.put(String.valueOf(c), pred);
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
	}
	
	@Override
	protected void doInit() {
		super.doInit();
	}
	
	@Override
	protected void doControl()
	{		
		if (generation >= maxOfGenerations) {
			//Get genotype of best individual
			String bestGenotype = ((StringTreeIndividual)bselector.select(bset, 1).get(0)).getGenotype();
			
			//Get base learner
			switch (classifierType) {
			case BR:
				learner = new BinaryRelevance(new J48());
				break;
				
			case LP:
				learner = new LabelPowerset2(new J48());
				((LabelPowerset2)learner).setSeed(seed);
				break;
			
			case CC:
				learner = new ClassifierChain(new J48(), utils.randomPermutation(fullTrainData.getNumLabels(), randgen));
				break;
				
			case PS:
				learner = new PrunedSets();
				break;
				
			case kLabelset:
				learner = null;
				break;

			default:
				learner = null;
				break;
			}
			
			//Generate ensemble object
			EMLC ensemble = new EMLC(learner, bestGenotype, useConfidences);
			
			//Print the leaves; i.e., different classifiers used in the ensemble
			System.out.println(utils.getLeaves(bestGenotype));
			
			try {
				//Build the ensemble
				ensemble.build(fullTrainData);
				
				//After building the ensemble, we can remove all the classifiers built and stored in hard disk
				utils.purgeDirectory(new File("mlc/"));
				
				//Evaluate with test data
				List<Measure> measures = prepareMeasures(fullTrainData);
				Evaluation results = new Evaluation(measures, fullTrainData);
				mulan.evaluation.Evaluator eval = new mulan.evaluation.Evaluator();
				results = eval.evaluate(ensemble, testData, measures);
				System.out.println(results);
				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			state = FINISHED;
			return;
		}
	}	
	
	/**
	 * Prepare measures for evaluation
	 * 
	 * @param data Multi-label dataset
	 * @return List of measures
	 */
	private List<Measure> prepareMeasures(MultiLabelInstances data) {
        List<Measure> measures = new ArrayList<Measure>();
        // add example-based measures
        measures.add(new HammingLoss());
        measures.add(new ModHammingLoss());
        measures.add(new SubsetAccuracy());
        measures.add(new ExampleBasedPrecision());
        measures.add(new ExampleBasedRecall());
        measures.add(new ExampleBasedFMeasure());
        measures.add(new ExampleBasedAccuracy());
        measures.add(new ExampleBasedSpecificity());
        // add label-based measures
        int numOfLabels = data.getNumLabels();
        measures.add(new MicroPrecision(numOfLabels));
        measures.add(new MicroRecall(numOfLabels));
        measures.add(new MicroFMeasure(numOfLabels));
        measures.add(new MicroSpecificity(numOfLabels));
	    measures.add(new MacroPrecision(numOfLabels));
	    measures.add(new MacroRecall(numOfLabels));
	    measures.add(new MacroFMeasure(numOfLabels));
	    measures.add(new MacroSpecificity(numOfLabels));
      
	    // add ranking-based measures if applicable
	    // add ranking based measures
	    measures.add(new AveragePrecision());
	    measures.add(new Coverage());
	    measures.add(new OneError());
	    measures.add(new IsError());
	    measures.add(new ErrorSetSize());
	    measures.add(new RankingLoss());
      
	    // add confidence measures if applicable
	    measures.add(new MeanAveragePrecision(numOfLabels));
	    measures.add(new GeometricMeanAveragePrecision(numOfLabels));
	    measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
	    measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
	    measures.add(new MicroAUC(numOfLabels));
	    measures.add(new MacroAUC(numOfLabels));

	    return measures;
    }
	
}
