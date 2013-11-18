package com.neuralnetwork.xor;

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
        return vWeights.dot(input)
                + vWeights.last(); //don't forget the bias
    }

    public float output(NVector input)
    {
        return phi.apply(rawoutput(input));
    }
}
