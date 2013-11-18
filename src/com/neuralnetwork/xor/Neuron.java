package com.neuralnetwork.xor;

/**
 * By convention, the bias is at the _end_ of the weights list
 *
 */
public class Neuron
{
    NVector vWeights;
    IActivationFunction phi;

    public Neuron(IActivationFunction phi, NVector vWeights)
    {
        this.vWeights = vWeights;
        this.phi = phi;
    }

    public Neuron(IActivationFunction phi, float... aWeights)
    {
        this.vWeights = new NVector(aWeights);
        this.phi = phi;
    }

    public float rawoutput(NVector input)
    {
        return vWeights.dot(input);
    }

    public float output(NVector input)
    {
        return phi.apply(rawoutput(input));
    }

    public float getWeight(int weight)
    {
        return vWeights.get(weight);
    }

    public int getNumberOfWeights()
    {
        return vWeights.size();
    }

    public IActivationFunction phi()
    {
        return phi;
    }

    public NVector getWeights()
    {
        return vWeights;
    }

    public void setWeight(int weight, float newWeight)
    {
        vWeights.set(weight, newWeight);
    }
}
