package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.ICnnMap;

import java.util.ArrayList;
import java.util.HashMap;

public class CNNBuilder {

    ArrayList<CNNLayer> layers = new ArrayList<CNNLayer>();

    private HashMap<ICnnMap,MapNode> hsMapInfos = new HashMap<ICnnMap,MapNode>();


    /**
     * Create {@link CNNLayer} from {@link CNNConnection}s
     *
     * @param connections
     * @return
     */
    public CNNBuilder setLayer(CNNConnection... connections)
    {
        CNNLayer layer = new CNNLayer();

        this.layers.add(layer);

        for(CNNConnection conn:connections)
        {
            MapNode mapNode = new MapNode(conn.map);
            hsMapInfos.put(conn.map, mapNode);

            for(ICnnMap inputMap:conn.lInputMaps)
            {
                MapNode inputMapNode = hsMapInfos.get(inputMap);

                if (inputMapNode == null)
                    throw new IllegalArgumentException("Input map must exist in previous layer");

                mapNode.addInput(inputMapNode);
            }

            layer.lMaps.add(mapNode);
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

            for(MapNode inputMap:cur.getInputMaps())
            {
                if (!prev.lMaps.contains(inputMap))
                    throw new IllegalArgumentException(
                            "Maps must be connected to maps in previous layer. " +
                                    "Failure in layer "+i);
            }

            for(MapNode map:cur.lMaps)
            {
                map.validateInputs();
            }
        }
    }
}
