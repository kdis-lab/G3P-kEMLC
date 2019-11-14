package gpemlc;

import java.util.Arrays;
import java.util.Comparator;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractEvaluator;
import net.sf.jclec.base.AbstractParallelEvaluator;
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
	 * Serialization constant
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
	 * Array of classifiers
	 */
	MultiLabelLearner[] classifiers;
	
	Utils utils = new Utils();
	
	/**
	 * Constructor
	 */
	public Evaluator()
	{
		super();
	}
	

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}
	
	public void setFullTrainData(MultiLabelInstances fullTrainData) {
		this.fullTrainData = fullTrainData;
	}
	
	public void setClassifiers(MultiLabelLearner[] classifiers) {
		this.classifiers = classifiers;
	}
	
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		//Individual genotype (list)
		String gen = ((StringTreeIndividual)ind).getGenotype();
		
		Utils utils = new Utils();
		double fitness = utils.countLeaves(gen);
		
		System.out.println("gen: " + gen);
		System.out.println("child: " + Arrays.toString(utils.getChildren(gen)));
		combine(utils.getChildren(gen));
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));

	}

	protected byte[][] combine(String[] nodes){
		byte[][] preds = new byte[fullTrainData.getNumInstances()][fullTrainData.getNumLabels()];

		for(String node : nodes) {
			if(utils.isLeaf(node)) {
				preds = sum(preds, getNodePredictions(Integer.parseInt(node)));
				System.out.println("leaf: " + node);
			}
			else {
				System.out.println("node: " + node);
				preds = sum(preds, combine(utils.getChildren(node)));
			}
		}
		

		return preds;
	}
	
	protected byte[][] getNodePredictions(int node){
		byte[][] preds = new byte[fullTrainData.getNumInstances()][fullTrainData.getNumLabels()];
		MultiLabelOutput mlo;
		
		try {
			for(int i=0; i<fullTrainData.getNumInstances(); i++) {
				mlo = classifiers[node].makePrediction(fullTrainData.getDataSet().get(i));
				for(int l=0; l<fullTrainData.getNumLabels(); l++) {
					if(mlo.getBipartition()[l]) {
						preds[i][l] ++;
					}
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return preds;
	}
	
	protected byte[][] sum(byte[][] b1, byte[][] b2){
		int nRow = b1.length, nCol = b1[0].length;
		byte[][] res = new byte[nRow][nCol];
		
		for(int i=0; i<nRow; i++) {
			for(int j=0; j<nRow; j++) {
				res[i][j] = (byte) (b1[i][j] + b2[i][j]);
			}
		}
		
		return res;
	}
	
}
