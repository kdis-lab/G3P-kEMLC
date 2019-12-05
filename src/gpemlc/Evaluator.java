package gpemlc;

import java.sql.Timestamp;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import gpemlc.utils.Utils;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractParallelEvaluator;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the evaluator for StringTreeIndividuals.
 * The tree is reduced, combining predictions in each node, and then final prediction is evaluated.
 * 
 * @author Jose M. Moyano
 *
 */
public class Evaluator extends AbstractParallelEvaluator {

	/**
	 * serialVersionUID
	 */
	private static final long serialVersionUID = 3488129084072488337L;

	/**
	 *  Indicates if fitness is to maximize 
	 */
	private boolean maximize = true;
	
	/**
	 *  Fitness comparator 
	 */
	private Comparator<IFitness> COMPARATOR = new ValueFitnessComparator(!maximize);
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Utils
	 */
	Utils utils = new Utils();
	
	/**
	 * Table with predictions of each classifier over fullTrainData
	 */
	Hashtable<String, Prediction> tablePredictions = new Hashtable<String, Prediction>();
	
	/**
	 * Indicates if confidences are used instead of bipartitions to combine predictions
	 */
	boolean useConfidences;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Locker for critical code blocks in parallel execution
	 */
	ReentrantLock lock = new ReentrantLock(); 
	
	/**
	 * Constructor
	 */
	public Evaluator()
	{
		super();
		utils = new Utils();
	}
	

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}
	
	/**
	 * Setter for fullTrainData
	 * 
	 * @param fullTrainData Full training data
	 */
	public void setFullTrainData(MultiLabelInstances fullTrainData) {
		this.fullTrainData = fullTrainData;
	}

	/**
	 * Setter for tablePredictions
	 * 
	 * @param tablePredictions Table with predictions of each classifier over full train data
	 */
	public void setTablePredictions(Hashtable<String, Prediction> tablePredictions) {
		this.tablePredictions = (Hashtable<String, Prediction>) tablePredictions;
	}
	
	/**
	 * Setter for useConfidences
	 * 
	 * @param useConfidences true if confidences are used instead of bipartitions to combine predictions
	 */
	public void setUseConfidences(boolean useConfidences) {
		this.useConfidences = useConfidences;
	}
	
	/**
	 * Setter for randgen
	 * 
	 * @param randgen Randgen
	 */
	public void setRandgen(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		//Lock here the use of randgen to generate the unique key for the individual
		lock.lock();
		String key = Long.toString(new Timestamp(System.currentTimeMillis()).getTime());
		int r = randgen.choose(3, 10);
		for(int i=0; i<r; i++) {
			key += (char)randgen.choose(48, 123);
		}
		lock.unlock();
		
		String gen = ((StringTreeIndividual)ind).getGenotype();
		System.out.println(gen);

		double fitness = 0.0;
		
		//Get final predictions by reducing the tree
		Prediction pred = reduce(gen,key);
		
		//Calculate fitness (ExF) with the reduced predictions
		fitness = exF(pred, fullTrainData);
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));
		System.out.println("f: " + fitness);
	}
	
	/**
	 * Calculate the Example-FMeasure (ExF) for given prediction of all instances and ground truth
	 * 
	 * @param pred Predictions
	 * @param mlData Multi-label data
	 * @return Example-based FMeasure
	 */
	protected double exF(Prediction pred, MultiLabelInstances mlData) {
		int[] labelIndices = mlData.getLabelIndices();
		boolean [] ground = new boolean[mlData.getNumLabels()];
		
		double exF = 0.0;
		
		for(int i=0; i<mlData.getNumInstances(); i++) {
			for(int j=0; j<mlData.getNumLabels(); j++) {
				//Get boolean ground truth for all labels in i-th instance
				if(mlData.getDataSet().get(i).value(labelIndices[j]) >= 0.5) {
					ground[j] = true;
				}
				else {
					ground[j] = false;
				}
			}
			
			//Calculate exF for prediction (as boolean bipartition) and ground truth of i-th instance
			exF += exFInstance(utils.confidenceToBipartition(pred.pred[i], 0.5), ground);
		}
		
		//Divide the exF by the number of instances and return it
		return exF/mlData.getNumInstances();
	}
	
	
	/**
	 * Calculate the Example-FMeasure (ExF) for given prediction and ground truth for one instance
	 * 
	 * @param pred Prediction of one instance
	 * @param ground Ground truth
	 * @return FMeasure for the instance
	 */
	protected double exFInstance(boolean[] pred, boolean[] ground) {
		int tp=0, fp=0, fn=0;
		
		for(int i=0; i<pred.length; i++) {
			if(pred[i] && ground[i]) {
				tp++;
			}
			else if(pred[i] && !ground[i]) {
				fp++;
			}
			else if(!pred[i] && ground[i]) {
				fn++;
			}
		}
		
		return InformationRetrievalMeasures.fMeasure(tp, fp, fn, 1.0);
	}
	
	/**
	 * Reduce the tree and obtain the final tree prediction
	 * 
	 * @param ind Individual, tree
	 * @return Final tree prediction
	 */
	public Prediction reduce(String ind, String key) {
		//Match two or more leaves (numer or _number) between parenthesis
		Pattern pattern = Pattern.compile("\\((_?\\d+ )+_?\\d+\\)");
		Matcher m = pattern.matcher(ind);
		
		//count to add predictions of combined nodes into the table
		int count = 0;
		Prediction pred = null;// = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
		while(m.find()) {
			//Combine the predictions of current nodes
			pred = combine(m.group(0), key);
			
			//Put combined predictions into the table
			tablePredictions.put(key+"_"+count, pred);
			
			//Replace node in the genotype
			ind = ind.substring(0, m.start()) + "_" + count + ind.substring(m.end(), ind.length());
			
			//Increment counter and match next one
			count++;
			m = pattern.matcher(ind);
		}
		
		//Return final prediction
		//	It is the combination of all nodes in level 1
		return pred;
	}
	
	/**
	 * Combine the predictions of several nodes
	 * 
	 * @param nodes String with the nodes to combine
	 * @return Combined prediction
	 */
	protected Prediction combine(String nodes, String key){
		Prediction pred = null; //new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
		Pattern pattern = Pattern.compile("\\d+");
		Matcher m;
		
		int n, nPreds = 0;
		
		//Split the nodes by space, so get the leaves
		String [] pieces = nodes.split(" ");
		for(String piece : pieces) {
			//Get index included in the piece and store in n
			m = pattern.matcher(piece);
			m.find();
			n = Integer.parseInt(m.group(0));

			//If the piece contains the character "_" is a combination of other nodes
			if(piece.contains("_")) {
				if(nPreds <= 0) {
					//If it is the first predictions to get, create new Prediction object with them 
					pred = new Prediction(tablePredictions.get(key+"_"+n));
				}
				else {
					//Add to the current prediction, the prediction of one of the previously combined nodes
					pred.addPrediction(tablePredictions.get(key+"_"+n));
				}
				
				//Remove because it is not going to be used again
				tablePredictions.remove(key+"_"+n);
				nPreds++;
			}
			else {
				if(nPreds <= 0) {
					//If it is the first predictions to get, create new Prediction object with them 
					pred = new Prediction(tablePredictions.get(String.valueOf(n)));
				}
				else {
					//Add to the current prediction, the prediction of the corresponding classifier
					pred.addPrediction(tablePredictions.get(String.valueOf(n)));
				}
				nPreds++;
			}
		}
		
		//Divide prediction by the number of learners and apply threshold if applicable
		if(useConfidences) {
			pred.divide();
		}
		else {
			pred.divideAndThresholdPrediction(0.5);
		}

		//Return combined prediction of corresponding nodes
		return pred;
	}
	
}
