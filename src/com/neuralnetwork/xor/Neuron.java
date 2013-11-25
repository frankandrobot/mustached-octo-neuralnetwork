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

    public Neuron(IActivationFunction phi, double... aWeights)
    {
        this.vWeights = new NVector(aWeights);
        this.phi = phi;
    }

    public double rawoutput(NVector input)
    {
        return vWeights.dot(input);
    }

    public double output(NVector input)
    {
        return phi.apply(rawoutput(input));
    }

    public double getWeight(int weight)
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

    public void setWeight(int weight, double newWeight)
    {
        vWeights.set(weight, newWeight);
    }

    @Override
    public String toString()
    {
        String rslt = String.format("%6.6g", vWeights.get(0));
        for(int i=1; i<vWeights.size(); ++i)
        {
            rslt += "  "+String.format("%6.6g", vWeights.get(i));
        }
        return rslt;
    }
}
