package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import com.neuralnetwork.nn.MultiLayerNN;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class NNBackprop
{
    DenseMatrix64F[] aMomentumTerms;

    Example[] examples;

    MultiLayerNN net;

    double eta;
    double gamma;

    NNBackpropHelper backprop;

    ErrorFunction error;

    NNBackprop(NNBackpropBuilder builder)
    {
        this.examples = builder.examples;

        this.net = builder.net;

        this.eta = builder.eta;
        this.gamma = builder.gamma;

        aMomentumTerms = new DenseMatrix64F[net.getLayers().length];

        for(int i=0; i<aMomentumTerms.length; i++)
        {
            DenseMatrix64F weights = net.getLayers()[i].getWeightMatrix();

            aMomentumTerms[0] = new DenseMatrix64F(weights.numRows, weights.numCols);
        }

        backprop = new NNBackpropHelper(net.getLayers());
    }

    private void reset()
    {
        for (DenseMatrix64F momemntum : aMomentumTerms)
        {
            CommonOps.fill(momemntum, 0);
        }
    }

    public void go(double epsilon)
    {
        reset();

        while(error.calculate(net, examples) >= epsilon)
        {
            backprop.resetCumulativeLearningTerms();

            for (Example example : examples)
            {
                backprop
                        .init(example)
                        .forwardProp()
                        .backprop()
                        .updateCumulativeLearningTerms();
            }

            addLearningTerms();
            addMomentumTerms();
            updateMomemtumTerms();
        }
    }

    private void addLearningTerms()
    {
        INeuralLayer[] layers = net.getLayers();

        DenseMatrix64F[] aLearningTerms = backprop.getCumulativeLearningTermsMinusEta();

        for(int i=0; i<layers.length; i++)
        {
            DenseMatrix64F weights = layers[i].getWeightMatrix();
            DenseMatrix64F learningTerms = aLearningTerms[i];

            for(int k=0; k<weights.data.length; k++)
            {
                weights.data[k] += eta * learningTerms.data[k];
            }
        }
    }

    private void addMomentumTerms()
    {
        INeuralLayer[] layers = net.getLayers();

        for(int i=0; i<layers.length; i++)
        {
            DenseMatrix64F weights = layers[i].getWeightMatrix();
            DenseMatrix64F momemtumTerms = aMomentumTerms[i];

            for(int k=0; k<weights.data.length; k++)
            {
                weights.data[k] += gamma * momemtumTerms.data[k];
            }
        }
    }

    private void updateMomemtumTerms()
    {
        INeuralLayer[] layers = net.getLayers();

        for(int i=0; i<layers.length; i++)
        {
            DenseMatrix64F weights = layers[i].getWeightMatrix();
            DenseMatrix64F momemtumTerms = aMomentumTerms[i];

            for(int k=0; k<weights.data.length; k++)
            {
                momemtumTerms.data[k] = weights.data[k];
            }
        }
    }
}
