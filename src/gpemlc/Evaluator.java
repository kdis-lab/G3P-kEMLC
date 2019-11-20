package gpemlc;

import java.util.Comparator;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import mulan.data.MultiLabelInstances;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractEvaluator;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;
import net.sf.jclec.stringtree.StringTreeIndividual;

/**
 * Class implementing the evaluator for individuals (MultipListIndividuals).
 * Each individual is transformed to the corresponding multi-label classifier and then evaluated.
 * 
 * @author Jose M. Moyano
 *
 */
public class Evaluator extends AbstractEvaluator {

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
		this.tablePredictions = tablePredictions;
	}
	
	/**
	 * Setter for useConfidences
	 * 
	 * @param useConfidences true if confidences are used instead of bipartitions to combine predictions
	 */
	public void setUseConfidences(boolean useConfidences) {
		this.useConfidences = useConfidences;
	}
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		String gen = ((StringTreeIndividual)ind).getGenotype();

		double fitness = 0.0;
		
		//Get predictions by reducing the ree
		Prediction pred = reduce(gen);
		
		//Calculate fitness (ExF) with the reduced predictions
		fitness = exF(pred, fullTrainData);
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));
	}
	
	/**
	 * Calculate the Example-FMeasure (ExF) for given prediction of all instances and true dataset
	 * 
	 * @param pred Predictions
	 * @param mlData Multi-label data
	 * @return ExF
	 */
	protected double exF(Prediction pred, MultiLabelInstances mlData) {
		int[] labelIndices = mlData.getLabelIndices();
		boolean [] ground = new boolean[mlData.getNumLabels()];
		
		double exF = 0.0;
		
		for(int i=0; i<mlData.getNumInstances(); i++) {
			for(int j=0; j<mlData.getNumLabels(); j++) {
				//Get ground truth for all labels in i-th instance
				if(mlData.getDataSet().get(i).value(labelIndices[j]) >= 0.5) {
					ground[j] = true;
				}
				else {
					ground[j] = false;
				}
			}
			
			//Calculate exF for prediction and ground truth of i-th instance
//			exF += exFInstance(pred.getBipartition(i, 0.5), ground);
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
	 * @return ExF
	 */
	protected double exFInstance(boolean[] pred, boolean[] ground) {
		double num=0, den=0;
		
		for(int i=0; i<pred.length; i++) {
			if(pred[i]) {
				den++;
			}
			if(ground[i]) {
				den++;
			}
			if(pred[i] && ground[i]) {
				num += 2;
			}
		}
		
		return num/den;
	}
	
	/**
	 * Reduce the tree and obtain the final tree prediction
	 * 
	 * @param ind Individual, tree
	 * @return Final tree prediction
	 */
	public Prediction reduce(String ind) {
		//Match two or more leaves (numer or _number) between parenthesis
		Pattern pattern = Pattern.compile("\\((_?\\d+ )+_?\\d+\\)");
		Matcher m = pattern.matcher(ind);
		
		//count to add predictions of combined nodes into the table
		int count = 0;
		Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
		while(m.find()) {
			//Combine the predictions of current nodes
			pred = combine(m.group(0));
			
			//Put combined predictions into the table
			tablePredictions.put("_"+count, pred);
			
			//Replace node in the genotype
			ind = ind.substring(0, m.start()) + "_" + count + ind.substring(m.end(), ind.length());
			
			//Increment counter and match next one
			count++;
			m = pattern.matcher(ind);
		}
		
		return pred;
	}
	
	/**
	 * Combine the predictions of several nodes
	 * 
	 * @param nodes String with the nodes to combine
	 * @return Combined prediction
	 */
	protected Prediction combine(String nodes){
		Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
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
				//Add to the current prediction, the prediction of one of the previously combined nodes
				pred.addPrediction(tablePredictions.get("_"+n));
				
				//Remove because it is not going to be used again
				tablePredictions.remove("_"+n);
				nPreds++;
			}
			else {
				//Add to the current prediction, the prediction of the corresponding classifier
				pred.addPrediction(tablePredictions.get(String.valueOf(n)));
				nPreds++;
			}
		}
		
		//Divide prediction by the number of learners and apply threshold
		if(useConfidences) {
			pred.divide(nPreds);
		}
		else {
			pred.divideAndThresholdPrediction(nPreds, 0.5);
		}

		return pred;
	}
	
}
