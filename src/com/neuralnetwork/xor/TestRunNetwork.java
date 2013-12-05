package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Random;

/**
 * Based on the network from
 * http://msdn.microsoft.com/en-us/magazine/jj658979.aspx
 */
public class TestRunNetwork
{
    /**
     * The layers indexed by training example and layer
     */
    LayorInfo[][] aExampleLayers;
    NVector[] aInput;
    NVector[] aExpected;

    NVector vError;

    protected static double alpha = 0.04; /** momentum **/
    protected static double eta = 0.9; /** learning parameter **/

    IActivationFunction.IDifferentiableFunction phi;
    protected SingleLayorNeuralNetwork firstPass;
    protected SingleLayorNeuralNetwork secondPass;

    public TestRunNetwork()
    {
        phi = new IActivationFunction.SigmoidUnityFunction();

        firstPass = new SingleLayorNeuralNetwork();
        secondPass = new SingleLayorNeuralNetwork();

        Random r = new Random();

        firstPass.setNeurons(new Neuron(phi, 0.10, 0.20, 0.30, -2.0),
                             new Neuron(phi, 0.40, 0.50, 0.60, -6.0),
                             new Neuron(phi, 0.70, 0.80, 0.90, 1.0),
                             new Neuron(phi, 1.00, 1.10, 1.20, -7.0));

        secondPass.setNeurons(new Neuron(phi, 1.30, 1.40, 1.5, 1.6, -2.5),
                              new Neuron(phi, 1.7, 1.8, 1.9, 2.0, -5.0));
   }

    protected void initializeExampleLayer(int layer)
    {
        aExampleLayers[layer] = new LayorInfo[2];

        aExampleLayers[layer][0] = new LayorInfo(firstPass);
        aExampleLayers[layer][1] = new LayorInfo(secondPass);
    }

    /**
     * This class wraps the SingleLayorNeuralNetwork
     * with additional information.
     *
     * Basically it works per layer
     * and you get back info about its individual neurons
     */
    protected class LayorInfo
    {
        final public SingleLayorNeuralNetwork layer;
        /**
         * For each neuron k in the layer,
         * store v_k (induced local field).
         */
        public NVector vInducedLocalField;
        /**
         * For each neuron k in the layer,
         * store y_k (value of impulse function).
         */
        public NVector vImpulseFunction;
        /**
         * Store the inputs from the previous layer.
         * vInput[i] is matched with a neuron's ith weight w_i
         * to produce the output i.e.,
         * vInput[i] = y^(l-1)_i
         */
        public NVector vInput;
        /**
         * For each neuron k in the layer,
         * aPrevWeights[k] is its previous set of weights
         */
        public NVector[] aPrevWeights;

        /**
         * For each neuron k in the layer,
         * store its gradient
         */
        public NVector vGradients;

        public LayorInfo(SingleLayorNeuralNetwork layer)
        {
            this.layer = layer;
            this.vInducedLocalField = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.vImpulseFunction = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.aPrevWeights = new NVector[this.layer.getNumberOfNeurons()];
            this.vGradients = new NVector().setSize(this.layer.getNumberOfNeurons());
        }
    }

    protected class Output
    {
        String weights()
        {
            String rslt = "";
            for(int neuron=0; neuron<firstPass.getNumberOfNeurons(); neuron++)
            {
                rslt += String.format("%20s | %20s %n", firstPass.aNeurons[neuron], getNeuron(secondPass, neuron));
            }
            return rslt;
        }

        private String getNeuron(SingleLayorNeuralNetwork network, int neuron)
        {
            return neuron < network.getNumberOfNeurons()
                    ? network.aNeurons[neuron].toString()
                    : "";
        }
    }

    public void backpropagation(double error, NVector... aInputExpected)
    {
        backpropagation(aInputExpected);

        int len = 1;

        while(vError.sumOfCoords() > error && len <= 1000)
        {
            System.out.println("====================================================================");
            System.out.println("Iteration = "+len++);
            System.out.format("Weights =%n%s", new Output().weights());
            System.out.format("Error = %s %n", vError.sumOfCoords());
            System.out.format("%5s %20s %20s%n", "i", "Expected", "Actual");
            for(int i=0; i<aExampleLayers.length; ++i)
                System.out.format("%5s %20s %20s%n",
                                  i,
                                  aExpected[i],
                                  aExampleLayers[i][aExampleLayers[i].length-1].vImpulseFunction);
            backpropagation(aInputExpected);
        }
    }

    /**
     * Trains the network using pairs of inputs/expected values
     *
     * @param aInputExpected list of input/expected vectors
     */
    protected void backpropagation(NVector... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        //initialization

        aExampleLayers = new LayorInfo[aInputExpected.length / 2][];
        aInput = new NVector[aExampleLayers.length];
        aExpected = new NVector[aExampleLayers.length];
        vError = new NVector().setSize(aExampleLayers.length);

        for(int i=0; i<aExampleLayers.length; i++)
        {
            initializeExampleLayer(i);
            aInput[i] = aInputExpected[2*i];
            aExpected[i] = aInputExpected[2*i+1];
        }

        for(int i=0; i<aExampleLayers.length; i++)
        {
            constructErrorFunction(i);
        }

        for(int i=0; i<aExampleLayers.length; i++)
        {
            constructGradients(i);
        }

        for(int i=0; i<aExampleLayers.length; i++)
        {
            adjustWeights(i);
        }
    }

    protected void constructErrorFunction(int example)
    {
        NVector actual = output(example, 0, aInput[example]);
        vError.set(example, aExpected[example].subtract(actual).dotProduct());
    }

    /**
     * Find the actual output for the given example.
     *
     * Side effect: store stuff (induced local field, impulse function, etc) in the layer info
     *
     * @param example index to example in aExampleLayers
     * @param layer the example's layer
     * @return output
     */
    protected NVector output(int example, int layer, NVector input)
    {
        LayorInfo layorInfo = aExampleLayers[example][layer];

        //save info for back propagation
        layorInfo.vInput = new NVector(input, 1f); //tack on bias at end
        constructInducedLocalField(layorInfo);
        constructImpulseFunction(layorInfo);

        if (layer < aExampleLayers[example].length - 1)
        {
            return output(example, layer+1, aExampleLayers[example][layer].vImpulseFunction);
        }
        return layorInfo.vImpulseFunction;
    }

    private void constructInducedLocalField(LayorInfo layorInfo)
    {
        //calculate v_k's
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vInducedLocalField.set(len++, neuron.rawoutput(layorInfo.vInput));
        }
    }

    private void constructImpulseFunction(LayorInfo layorInfo)
    {
        //calculate y_k's aka output
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vImpulseFunction.set(len, neuron.phi().apply(layorInfo.vInducedLocalField.get(len++)));
        }
    }

    protected void constructGradients(int example)
    {
       constructGradients(example, aExampleLayers[example].length - 1);
    }

    protected void constructGradients(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aExampleLayers[example][layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                layorInfo.vGradients.set(neuronPos, gradient(example, layer, neuronPos));

                neuronPos++;
            }

            constructGradients(example, layer - 1);
        }
    }

    /**
     * Given a network layor and the neuron's position (in the array)
     * return its gradient.
     *
     * @param layerLevel
     * @param neuron
     * @return
     */
    protected double gradient(int example, int layerLevel, int neuron)
    {
        if (layerLevel == aExampleLayers[example].length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            double actualOutput = aExampleLayers[example][layerLevel].vInducedLocalField.get(neuron);
            return (aExpected[example].get(neuron) - actualOutput)
                    * phi.derivative(actualOutput);
        }
        else
        {
             return phi.derivative( aExampleLayers[example][layerLevel].vInducedLocalField.get(neuron))
                     * sumGradients(example, layerLevel + 1, neuron);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron
     *
     * @param layerLevel layer level
     * @param currentNeuron current neuron
     * @return
     */
    protected double sumGradients(int example, int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aExampleLayers[example][layerLevel];
        double rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(example, layerLevel, neuronPos++);
        }
        return rslt;
    }

    protected void adjustWeights(int example)
    {
        adjustWeights(example, aExampleLayers[example].length - 1);
    }

    protected void adjustWeights(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aExampleLayers[example][layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                if (layorInfo.aPrevWeights[neuronPos] == null)
                    layorInfo.aPrevWeights[neuronPos] = new NVector().setSize(neuron.getNumberOfWeights());

                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    double newWeight = neuron.getWeight(weight)
                            + alpha * layorInfo.aPrevWeights[neuronPos].get(weight) //momentum
                            + eta * layorInfo.vGradients.get(neuronPos) * layorInfo.vInput.get(weight); //delta correction

                    //backup previous weight
                    layorInfo.aPrevWeights[neuronPos].set(weight, neuron.getWeight(weight));

                    neuron.setWeight(weight, newWeight);
                }

                neuronPos++;
            }

            adjustWeights(example, layer - 1);
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
