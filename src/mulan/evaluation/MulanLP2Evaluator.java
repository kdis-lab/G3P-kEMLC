package mulan.evaluation;

import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;

public class MulanLP2Evaluator extends Evaluator {

	public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException, Exception {
		((LabelPowerset2)learner).setSeed(1);
		return super.evaluate(learner, data, measures);
	}
	
}
