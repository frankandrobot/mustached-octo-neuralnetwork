package com.neuralnetwork.core;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import com.neuralnetwork.core.interfaces.INeuron;
import com.neuralnetwork.core.neuron.NVector;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class MultiLayerNetworkBackprop<T,N extends INeuron>
{
    protected class OutputInfo
    {
        public T inducedLocalField;
        public T output;
    }

    protected class GradientInfo
    {
        public double[] gradients;
    }

    protected INeuralLayer<T,N>[] aLayers;

    protected OutputInfo[] aLayerInfo;
    protected GradientInfo[] aGradientInfo;

    /**
     * Aka ... error
     */
    protected NVector vTotalDifferenceSquared;

    final protected double alpha; /** momentum **/
    final protected double eta; /** learning parameter **/
    final protected int numberIterations;

    protected IActivationFunction.IDifferentiableFunction phi;

    public MultiLayerNetworkBackprop(Builder builder)
    {
        this.eta = builder.eta;
        this.alpha = builder.alpha;
        this.phi = builder.phi;
        this.numberIterations = builder.numberIterations == 0 ? 1000 : builder.numberIterations;

        initializeLayers(builder);
    }

    protected void setOutput(int index, T input)
    {
        aLayerInfo[index].inducedLocalField = aLayers[index].generateInducedLocalField(input);
        aLayerInfo[index].output = aLayers[index].generateOutput(input);
    }

    MultiLayerNetworkBackprop forwardProp(T input)
    {
        setOutput(0, input);

        for(int i=1; i<aLayers.length; i++)
        {
            setOutput(i, aLayerInfo[i - 1].output);
        }

        return this;
    }

    /**
     * Requires that you call forwardProp first
     *
     * @return
     */
    MultiLayerNetworkBackprop backprop()
    {
        for(int i=aLayers.length-1; i>=0 ;i--)
        {
            constructGradients(i);
        }

        return this;
    }

    protected void constructGradients(int layerIndex)
    {
        INeuralLayer<T,N> layer = aLayers[layerIndex];
        N[] aNeurons = layer.getNeurons();

        for(int neuronPos=0; neuronPos<aNeurons.length; neuronPos++)
        {
            aGradientInfo[layerIndex].gradients[0] = gradient(layer, neuronPos);
        }
    }

    /**
     * Gradient for the given example, layer, and neuron
     *
     * @param example example
     * @param layerLevel layer level
     * @param neuron neuron
     * @return gradient value for the given example, layer, and neuron
     */
    protected double gradient(int layerLevel, int neuron)
    {
        if (layerLevel == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = aLayerInfo[layerLevel].inducedLocalField.get(neuron);
            final double impulseFunction = aLayerInfo[layerLevel].output.get(neuron)
                    ;
            return (aExamples[example].vExpected.get(neuron) - impulseFunction)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative( aLayers[layerLevel].vInducedLocalField.get(neuron))
                     * sumGradients(example, layerLevel + 1, neuron);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron
     *
     *
     *
     * @param example
     * @param layerLevel layer level
     * @param currentNeuron current neuron
     * @return
     */
    protected double sumGradients(int example, int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aLayers[layerLevel];
        double rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(example, layerLevel, neuronPos++);
        }
        return rslt;
    }

    protected void saveWeightAdjustments()
    {
        saveWeightAdjustments(aLayers.length - 1);
    }

    protected void saveWeightAdjustments(int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    double weightAdjustment =
                            eta * layorInfo.vGradients.get(neuronPos) * layorInfo.vInput.get(weight); //delta correction

                    //save weight adjustment
                    double curWeightAdjustment = layorInfo.aWeightAdjustments[neuronPos].get(weight);
                    layorInfo.aWeightAdjustments[neuronPos].set(weight, curWeightAdjustment + weightAdjustment);
                }

                neuronPos++;
            }

            saveWeightAdjustments(layer - 1);
        }
    }

    protected void adjustWeights()
    {
        for(LayorInfo layorInfo:aLayers)
        {
            //iterate thru the neurons in the layer
            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    //adjust weights
                    double momentum = alpha * layorInfo.aPrevWeights[neuronPos].get(weight);
                    double curWeight = neuron.getWeight(weight);

                    //backup previous weight
                    layorInfo.aPrevWeights[neuronPos].set(weight, neuron.getWeight(weight));

                    neuron.setWeight(weight,
                                     curWeight
                                     + momentum
                                     + layorInfo.aWeightAdjustments[neuronPos].get(weight) // total delta weights
                    );
                }
                neuronPos++;
            }
        }
    }

    protected void resetWeightAdjustments()
    {
        for(LayorInfo layorInfo:aLayers)
        {
            //iterate thru the neurons in the layer
            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    layorInfo.aWeightAdjustments[neuronPos].set(weight, 0);
                }
                neuronPos++;
            }
        }
    }

    public NVector rawoutput(NVector input)
    {
        throw new NotImplementedException();
    }

//    @Override
//    public int getNumberOfNeurons()
//    {
//        return 3;
//    }


}
