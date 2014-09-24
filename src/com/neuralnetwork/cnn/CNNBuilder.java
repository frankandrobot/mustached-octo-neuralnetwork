package com.neuralnetwork.cnn;

import com.neuralnetwork.core.Dimension;
import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;

import java.util.ArrayList;
import java.util.HashMap;

public class CNNBuilder {

    ArrayList<CNNLayer> layers = new ArrayList<CNNLayer>();

    private HashMap<IMatrixNeuralLayer,MapInfo> hsMapInfos = new HashMap<IMatrixNeuralLayer,MapInfo>();


    public CNNBuilder setLayer(CNNConnection... connections)
    {
        CNNLayer layer = new CNNLayer();

        this.layers.add(layer);

        for(CNNConnection conn:connections)
        {
            MapInfo mapInfo = new MapInfo(conn.map);
            hsMapInfos.put(conn.map, mapInfo);

            for(IMatrixNeuralLayer inputMap:conn.lInputMaps)
            {
                MapInfo inputMapInfo = hsMapInfos.get(inputMap);

                if (inputMapInfo == null)
                    throw new IllegalArgumentException("Input map must exist in previous layer");

                mapInfo.addInput(inputMapInfo);
            }

            layer.lMaps.add(mapInfo);
        }

        return this;
    }

    public CNN build() throws Exception
    {
        validate();

        return new CNN(this);
    }

    private void validate() throws Exception
    {
        if (layers.get(0).getInputMaps().size() != 0)
            throw new IllegalArgumentException("Input layers do not have input maps");

        for(int i=1; i<layers.size(); i++)
        {
            CNNLayer cur = layers.get(i);
            CNNLayer prev = layers.get(i-1);

            for(MapInfo inputMap:cur.getInputMaps())
            {
                if (!prev.lMaps.contains(inputMap))
                    throw new IllegalArgumentException(
                            "Maps must be connected to maps in previous layer. " +
                                    "Failure in layer "+i);
            }

            for(MapInfo map:cur.lMaps)
            {
                Dimension inputDims = map.map.getInputDim();

                for(MapInfo inputMap:map.getInputMaps())
                {
                    Dimension outputDims = inputMap.map.getOutputDim();

                    if (!inputDims.equals(outputDims))
                        throw new IllegalArgumentException(
                                "Input dimensions must match." +
                                        "Failure in layer " + i);
                }
            }
        }
    }
}
