

async def ingest_products_data(client: Graphiti):
    script_dir = Path.cwd().parent
    json_file_path = script_dir / 'data' / 'manybirds_products.json'

    with open(json_file_path) as file:
        products = json.load(file)['products']

    for i, product in enumerate(products):
        await client.add_episode(
            name=product.get('title', f'Product {i}'),
            episode_body=str({k: v for k, v in product.items() if k != 'images'}),
            source_description='ManyBirds products',
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )


await ingest_products_data(client)