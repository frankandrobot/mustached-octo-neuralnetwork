package com.neuralnetwork.core.neuron;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuron;

/**
 * We arbitrarily put the bias at the *beginning* of the weights list
 *
 */
public class Neuron implements INeuron<double[]>
{
    IActivationFunction.IDifferentiableFunction phi;

    double[] weights;
    double[] weightsWithoutBias;

    public Neuron(IActivationFunction.IDifferentiableFunction phi, double... aWeights)
    {
        this.weights = aWeights;
        this.phi = phi;

        weightsWithoutBias = new double[weights.length - 1];
        System.arraycopy(weights,1,weightsWithoutBias,0,weights.length-1);
    }

    @Override
    public double getWeight(int weight)
    {
        return weights[weight];
    }

    @Override
    public void setWeight(int weight, double newWeight)
    {
        weights[weight] = newWeight;
        if (weight > 0) weightsWithoutBias[weight-1] = newWeight;
    }

    @Override
    public int getNumberOfWeights()
    {
        return weights.length;
    }

    /**
     * This should be read only!!!
     * @return
     */
    @Override
    public double[] getWeightsWithoutBias()
    {
        return weightsWithoutBias;
    }

    /**
     * This should be read only!!!!
     * @return
     */
    @Override
    public double[] getWeights()
    {
        return weights;
    }

    @Override
    public IActivationFunction.IDifferentiableFunction phi()
    {
        return phi;
    }

    @Override
    public String toString()
    {
        String rslt = String.format("%6.6g", weights[0]);
        for(int i=1; i< weights.length; ++i)
        {
            rslt += "  "+String.format("%6.6g", weights[i]);
        }
        return rslt;
    }
}
