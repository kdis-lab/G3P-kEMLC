package gpemlc;

import java.io.File;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.configuration.Configuration;

import gpemlc.mutator.Mutator;
import gpemlc.recombinator.Crossover;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.*;
import net.sf.jclec.algorithm.classic.SGE;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.stringtree.StringTreeCreator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;

public class Alg extends SGE {

	/**
	 * 
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
	 * Training datasets
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
	 * Number of MLC
	 */
	int nMLC;
	
	/**
	 * Utils
	 */
	Utils utils;
	
	/**
	 * Table with built classifiers
	 */
	Hashtable<String, MultiLabelLearnerBase> table;
	
	/**
	 * Table with predictions of each classifier
	 */
	Hashtable<String, Prediction> tablePredictions;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	@Override
	public void configure(Configuration configuration) {
		super.configure(configuration);
		
		randgen = randGenFactory.createRandGen();
		utils = new Utils(randgen);
		
		//Initialize tables
		table = new Hashtable<String, MultiLabelLearnerBase>();
		tablePredictions = new Hashtable<String, Prediction>();
		
		//Get datasets
		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
		nMLC = configuration.getInt("different-classifiers");
		maxChildren = configuration.getInt("max-children");
		maxDepth = configuration.getInt("max-depth");
		
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
			
			for(int c=0; c<nMLC; c++) {
				//Sample c-th data
				currentTrainData = MulanUtils.sampleData(fullTrainData, sampleRatio, randgen);
				
				//Build classifier with c-th data and store in the table
				LabelPowerset2 lp = new LabelPowerset2(new J48());
				lp.setSeed(1);
				table.put(String.valueOf(c), lp);
				table.get(String.valueOf(c)).build(currentTrainData);
				
				//Store object of classifier in the hard disk
				
				utils.writeObject(table.get(String.valueOf(c)), "mlc/classifier"+c+".mlc");
				
				//Get predictions of c-th classifier over all data
				Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
				for(int i=0; i<fullTrainData.getNumInstances(); i++) {
					boolean[] bip = table.get(String.valueOf(c)).makePrediction(fullTrainData.getDataSet().get(i)).getBipartition();
					for(int j=0; j<fullTrainData.getNumLabels(); j++) {
						if(bip[j]) {
							pred.bip[i][j] = 1;
						}
						else {
							pred.bip[i][j] = 0;
						}
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
	}
	
	@Override
	protected void doInit() {
		super.doInit();
	}
	
	@Override
	protected void doControl()
	{
//		System.out.println("Generation " + generation);
		
		if (generation >= maxOfGenerations) {
			LabelPowerset2 lp = new LabelPowerset2(new J48());
			lp.setSeed(1);
			String bestGenotype = ((StringTreeIndividual)bselector.select(bset, 1).get(0)).getGenotype();
			EMLC ensemble = new EMLC(lp, bestGenotype);
			System.out.println(utils.getLeaves(bestGenotype));
			
			try {
				ensemble.build(fullTrainData);
				
				//After building the ensemble, we can remove all the classifiers built and stored in hard disk
				utils.purgeDirectory(new File("mlc/"));
				
				List<Measure> measures = prepareMeasures(fullTrainData);
				Evaluation results = new Evaluation(measures, fullTrainData);
				mulan.evaluation.Evaluator eval = new mulan.evaluation.Evaluator();
				results = eval.evaluate(ensemble, testData, measures);
				System.out.println(results);
				
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			state = FINISHED;
			return;
		}
	}	
	
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
//      measures.add(new AveragePrecision());
//      measures.add(new Coverage());
//      measures.add(new OneError());
//      measures.add(new IsError());
//      measures.add(new ErrorSetSize());
//      measures.add(new RankingLoss());
      
      // add confidence measures if applicable
//      measures.add(new MeanAveragePrecision(numOfLabels));
//      measures.add(new GeometricMeanAveragePrecision(numOfLabels));
//      measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
//      measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
	    measures.add(new MicroAUC(numOfLabels));
//      measures.add(new MacroAUC(numOfLabels));

        return measures;
    }
	
}
