package com.neuralnetwork.cnn.old;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.old.OldConvolutionMapLayer;
import com.neuralnetwork.cnn.old.OldFeatureMap;
import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class OldConvolutionMapTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    @Test
    public void testConvolution()
    {
        final DenseMatrix64F input = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();

        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);
        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        featureMap.generateOutput(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(featureMap.mFeatureMap.numRows, is(2) );
        assertThat(featureMap.mFeatureMap.numCols, is(2) );

        double o11 = input.get(0,0)*weights[0] + input.get(0,1)*weights[1]
                + input.get(1,0)*weights[2] + input.get(1,1)*weights[3]
                + weights[4];
        o11 = phi.apply(o11);

        assertThat(featureMap.mFeatureMap.get(0, 0), is(o11));

        double o12 = input.get(0,1)*weights[0] + input.get(0,2)*weights[1]
                + input.get(1,1)*weights[2] + input.get(1,2)*weights[3]
                + weights[4];
        o12 = phi.apply(o12);

        assertThat(featureMap.mFeatureMap.get(0,1), is(o12));

        double o21 = input.get(1,0)*weights[0] + input.get(1,1)*weights[1]
                + input.get(2,0)*weights[2] + input.get(2,1)*weights[3]
                + weights[4];
        o21 = phi.apply(o21);

        assertThat(featureMap.mFeatureMap.get(1,0), is(o21));
    }

    @Test
    public void testConvolutionRawOutputFuncs()
    {
        final DenseMatrix64F input = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        DenseMatrix64F output = featureMap.calculateFeatureMap(input);

        assertThat(output.get(0,0), is(featureMap.output(input, 0, 0)));
        assertThat(output.get(1,0), is(featureMap.output(input, 1, 0)));
        assertThat(output.get(0,1), is(featureMap.output(input, 0, 1)));
        assertThat(output.get(1,1), is(featureMap.output(input, 1, 1)));

        final double o11 = input.get(0,0)*weights[0] + input.get(0,1)*weights[1]
                + input.get(1,0)*weights[2] + input.get(1,1)*weights[3]
                + weights[4];

        assertThat(featureMap.rawoutput(input,0,0), is(o11));

        final double o12 = input.get(0,1)*weights[0] + input.get(0,2)*weights[1]
                + input.get(1,1)*weights[2] + input.get(1,2)*weights[3]
                + weights[4];

        assertThat(featureMap.rawoutput(input, 0, 1), is(o12));

        final double o21 = input.get(1,0)*weights[0] + input.get(1,1)*weights[1]
                + input.get(2,0)*weights[2] + input.get(2,1)*weights[3]
                + weights[4];

        assertThat(featureMap.rawoutput(input,1,0), is(o21));
    }

    @Test
    public void testWeightConnections()
    {
        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.set1DInputSize(8);
        builder.setReceptiveFieldSize(3 * 3);

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        int[] aWeights = new int[featureMap.receptiveFieldSize];

        //test in the middle
        featureMap.disableWeightConnections(aWeights, 4, 4);
        for(int i=0; i<aWeights.length; i++)
            assertThat(aWeights[i], is(1));

        //test top border starting in top left
        featureMap.disableWeightConnections(aWeights, 0, 0);
        assertThat(aWeights[0], is(1));
        for(int i=1; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 0, 1);
        assertThat(aWeights[0], is(1));
        assertThat(aWeights[1], is(1));
        for(int i=2; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 0, 2);
        assertThat(aWeights[0], is(1));
        assertThat(aWeights[1], is(1));
        assertThat(aWeights[2], is(1));
        for(int i=3; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 0, 4);
        assertThat(aWeights[0], is(1));
        assertThat(aWeights[1], is(1));
        assertThat(aWeights[2], is(1));
        for(int i=3; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 0, 5);
        assertThat(aWeights[0], is(1));
        assertThat(aWeights[1], is(1));
        assertThat(aWeights[2], is(1));
        for(int i=3; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 0, 6);
        assertThat(aWeights[0], is(0));
        assertThat(aWeights[1], is(1));
        assertThat(aWeights[2], is(1));
        for(int i=3; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 0, 7);
        assertThat(aWeights[2], is(1));
        for(int i=0; i<aWeights.length; i++)
            if (i != 2) assertThat(aWeights[i], is(0));

        //next level
        featureMap.disableWeightConnections(aWeights, 1, 7);
        assertThat(aWeights[2], is(1));
        assertThat(aWeights[5], is(1));
        for(int i=0; i<aWeights.length; i++)
            if (i!=2 && i!= 5) assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 1, 6);
        assertThat(aWeights[0], is(0));
        assertThat(aWeights[1], is(1));
        assertThat(aWeights[2], is(1));
        assertThat(aWeights[3], is(0));
        assertThat(aWeights[4], is(1));
        assertThat(aWeights[5], is(1));
        for(int i=6; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 1, 5);
        for(int i=0; i<6; i++)
            assertThat(aWeights[i], is(1));
        for(int i=6; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 1, 2);
        for(int i=0; i<6; i++)
            assertThat(aWeights[i], is(1));
        for(int i=6; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 1, 1);
        assertThat(aWeights[0], is(1));
        assertThat(aWeights[1], is(1));
        assertThat(aWeights[2], is(0));
        assertThat(aWeights[3], is(1));
        assertThat(aWeights[4], is(1));
        assertThat(aWeights[5], is(0));
        for(int i=6; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        featureMap.disableWeightConnections(aWeights, 1, 0);
        assertThat(aWeights[0], is(1));
        assertThat(aWeights[1], is(0));
        assertThat(aWeights[2], is(0));
        assertThat(aWeights[3], is(1));
        assertThat(aWeights[4], is(0));
        assertThat(aWeights[5], is(0));
        for(int i=6; i<aWeights.length; i++)
            assertThat(aWeights[i], is(0));

        //test bottom border
        featureMap.disableWeightConnections(aWeights, 7, 2);
        for(int i=0; i<6; i++)
            assertThat(aWeights[i], is(0));
        for(int i=6; i<9; i++)
            assertThat(aWeights[i], is(1));
    }

    private double[] getWeights(int [] weights)
    {
        double[] rslt = new double[weights.length];
        for(int i=0; i<weights.length; i++)
            rslt[i] = weights[i];
        return rslt;
    }

    @Test
    public void testMapToFeatureMapRow()
    {
        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.set1DInputSize(8);
        builder.setReceptiveFieldSize(3 * 3);

        //0 1 2
        //3 4 5
        //6 7 8

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        for(int inputRow = 1; inputRow<10; inputRow++)
        {
            int row;

            row = featureMap.featureMapRowPosition(0, inputRow);
            assertThat(row, is(inputRow));

            row = featureMap.featureMapRowPosition(1, inputRow);
            assertThat(row, is(inputRow));

            row = featureMap.featureMapRowPosition(2, inputRow);
            assertThat(row, is(inputRow));

            row = featureMap.featureMapRowPosition(3, inputRow);
            assertThat(row, is(inputRow-1));

            row = featureMap.featureMapRowPosition(4, inputRow);
            assertThat(row, is(inputRow-1));

            row = featureMap.featureMapRowPosition(5, inputRow);
            assertThat(row, is(inputRow-1));

            row = featureMap.featureMapRowPosition(6, inputRow);
            assertThat(row, is(inputRow-2));

            row = featureMap.featureMapRowPosition(7, inputRow);
            assertThat(row, is(inputRow-2));

            row = featureMap.featureMapRowPosition(8, inputRow);
            assertThat(row, is(inputRow-2));
        }
    }

    @Test
    public void testMapToFeatureMapCol()
    {
        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.set1DInputSize(8);
        builder.setReceptiveFieldSize(3 * 3);

        //0 1 2
        //3 4 5
        //6 7 8

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        for(int inputCol = 1; inputCol<10; inputCol++)
        {
            int col;

            col = featureMap.featureMapColPosition(0, inputCol);
            assertThat(col, is(inputCol));

            col = featureMap.featureMapColPosition(1, inputCol);
            assertThat(col, is(inputCol-1));

            col = featureMap.featureMapColPosition(2, inputCol);
            assertThat(col, is(inputCol-2));

            col = featureMap.featureMapColPosition(3, inputCol);
            assertThat(col, is(inputCol));

            col = featureMap.featureMapColPosition(4, inputCol);
            assertThat(col, is(inputCol-1));

            col = featureMap.featureMapColPosition(5, inputCol);
            assertThat(col, is(inputCol-2));

            col = featureMap.featureMapColPosition(6, inputCol);
            assertThat(col, is(inputCol));

            col = featureMap.featureMapColPosition(7, inputCol);
            assertThat(col, is(inputCol-1));

            col = featureMap.featureMapColPosition(8, inputCol);
            assertThat(col, is(inputCol-2));
        }
    }
}
