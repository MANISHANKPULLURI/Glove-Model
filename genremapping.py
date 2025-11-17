def create_genre_mapping():
    genre_mapping = {}
    action_genres = [
        'action', 'action film', 'martial arts', 'superhero', 'action drama',
        'spy', 'spy film', 'swashbuckler', 'fight', 'combat', 'biker film',
        'superhero film', 'action thriller', 'action comedy', 'action-comedy',
        'spy action', 'biker', 'superhero action', 'combat film'
    ]
    adventure_genres = [
        'adventure', 'action adventure', 'expedition', 'road movie',
        'treasure hunt', 'survival', 'exploration', 'road thriller', 'quest',
        'journey', 'travel', 'wilderness'
    ]
    animation_genres = [
        'animation', 'animated', 'animated film', 'anime', 'clay animation',
        'stop-motion', 'cartoon', 'animated short', 'cgi animation',
        'live-action/animation', 'animation comedy family'
    ]
    biography_genres = [
        'biography', 'biographical', 'biopic', 'true story', 'docudrama',
        'biogtaphy', 'life story', 'biographical drama', 'historical biography'
    ]
    comedy_genres = [
        'comedy', 'romantic comedy', 'slapstick', 'parody', 'satire',
        'black comedy', 'sitcom', 'comedy drama', 'rom com', 'spoof',
        'comedy horror', 'comedy western', 'comedy action', 'dramedy'
    ]
    crime_genres = [
        'crime', 'detective', 'gangster', 'heist', 'police', 'noir',
        'true crime', 'crime drama', 'crime thriller', 'neo-noir',
        'mystery thriller', 'crime comedy'
    ]
    documentary_genres = [
        'documentary', 'educational', 'nature', 'reality', 'documentary drama',
        'mockumentary', 'docudrama', 'documentary film', 'investigative'
    ]
    drama_genres = [
        'drama', 'melodrama', 'tragedy', 'dramatic', 'period drama',
        'drama film', 'romantic drama', 'family drama', 'social drama',
        'psychological drama', 'character study'
    ]
    experimental_genres = [
        'experimental', 'avant-garde', 'art film', 'abstract', 'indie',
        'independent', 'student film', '16 mm film', 'art house',
        'experimental film', 'underground'
    ]
    family_genres = [
        'family', 'children', 'kids', 'family comedy', 'family drama',
        'family adventure', 'family film', 'juvenile', 'educational children'
    ]
    fantasy_genres = [
        'fantasy', 'fairy tale', 'magical', 'supernatural', 'mythological',
        'fantasy adventure', 'fantasy drama', 'epic fantasy', 'sword and sorcery'
    ]
    historical_genres = [
        'historical', 'period', 'costume drama', 'epic', 'classical',
        'period piece', 'historical drama', 'historical epic',
        'period film', 'shakespearean'
    ]
    horror_genres = [
        'horror', 'scary', 'supernatural horror', 'slasher', 'ghost',
        'monster', 'zombie', 'psychological horror', 'gore', 'splatter',
        'horror thriller', 'horror comedy', 'horror drama'
    ]
    musical_genres = [
        'musical', 'music', 'dance', 'opera', 'concert', 'musical comedy',
        'musical drama', 'rock musical', 'operetta', 'music documentary'
    ]
    mystery_genres = [
        'mystery', 'whodunit', 'detective story', 'puzzle', 'enigma',
        'mystery thriller', 'crime mystery', 'supernatural mystery',
        'mystery drama'
    ]
    romance_genres = [
        'romance', 'love story', 'romantic', 'romantic comedy',
        'romantic drama', 'rom-com', 'romance film', 'romantic fantasy',
        'love', 'relationship drama'
    ]
    scifi_genres = [
        'science fiction', 'sci-fi', 'sci fi', 'space', 'futuristic',
        'cyberpunk', 'space opera', 'science fantasy', 'time travel',
        'sci-fi action', 'sci-fi drama'
    ]
    short_genres = [
        'short', 'short film', 'short subject', 'short animation',
        'short drama', 'short documentary', 'student short'
    ]
    social_genres = [
        'social', 'political', 'propaganda', 'activism', 'social drama',
        'political thriller', 'social commentary', 'protest', 'issue film'
    ]
    sports_genres = [
        'sports', 'sports drama', 'sports comedy', 'athletic',
        'competition', 'racing', 'boxing', 'wrestling', 'olympic'
    ]
    thriller_genres = [
        'thriller', 'suspense', 'psychological thriller',
        'crime thriller', 'horror thriller', 'erotic thriller',
        'suspense thriller', 'mystery thriller'
    ]
    war_genres = [
        'war', 'military', 'war drama', 'war film', 'anti-war',
        'world war i', 'world war ii', 'civil war', 'p.o.w.',
        'war documentary', 'war action'
    ]
    western_genres = [
        'western', 'cowboy', 'western drama', 'spaghetti western',
        'wild west', 'western comedy', 'frontier', 'western action'
    ]
    world_genres = [
        'world cinema', 'foreign', 'international', 'bollywood',
        'european', 'asian', 'foreign language', 'international co-production'
    ]
    youth_genres = [
        'youth', 'teen', 'coming of age', 'teen comedy', 'teen drama',
        'young adult', 'high school', 'teen romance', 'adolescent'
    ]
    all_genre_lists = [
        ('Action', action_genres),
        ('Adventure', adventure_genres),
        ('Animation', animation_genres),
        ('Biography', biography_genres),
        ('Comedy', comedy_genres),
        ('Crime', crime_genres),
        ('Documentary', documentary_genres),
        ('Drama', drama_genres),
        ('Experimental', experimental_genres),
        ('Family', family_genres),
        ('Fantasy', fantasy_genres),
        ('Historical', historical_genres),
        ('Horror', horror_genres),
        ('Musical', musical_genres),
        ('Mystery', mystery_genres),
        ('Romance', romance_genres),
        ('Science Fiction', scifi_genres),
        ('Short Film', short_genres),
        ('Social', social_genres),
        ('Sports', sports_genres),
        ('Thriller', thriller_genres),
        ('War', war_genres),
        ('Western', western_genres),
        ('World Cinema', world_genres),
        ('Youth', youth_genres)
    ]
    for parent_genre, genre_list in all_genre_lists:
        for genre in genre_list:
            genre_mapping[genre.lower()] = parent_genre
    genre_mapping = add_compound_genres(genre_mapping)
    all_mappings = dict(genre_mapping)
    for genre, parent in list(genre_mapping.items()):
        all_mappings[f"{genre} film"] = parent
        all_mappings[genre.replace(' ', '-')] = parent
        all_mappings[genre.replace(' ', '/')] = parent
        all_mappings[f"{genre} movie"] = parent
    return all_mappings

def get_unique_genres_from_file(file_path):
    try:
        genre_mapping = create_genre_mapping()
        original_genres = set()
        mapped_genres = set()
        unmapped_genres = set()
        genre_to_parent = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    genres = parts[0].lower().split(',')
                    for genre in genres:
                        genre = cleanup_genre_name(genre)
                        if genre:
                            original_genres.add(genre)
                            if genre in genre_mapping:
                                parent = genre_mapping[genre]
                                mapped_genres.add(parent)
                                genre_to_parent[genre] = parent
                            else:
                                unmapped_genres.add(genre)
        genres_by_parent = {}
        for genre in original_genres:
            if genre in genre_to_parent:
                parent = genre_to_parent[genre]
                if parent not in genres_by_parent:
                    genres_by_parent[parent] = []
                genres_by_parent[parent].append(genre)
        return {
            'original_genres': sorted(list(original_genres)),
            'mapped_genres': sorted(list(mapped_genres)),
            'unmapped_genres': sorted(list(unmapped_genres)),
            'genres_by_parent': genres_by_parent
        }
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def create_mapped_dataset(input_filepath, output_filepath):
    genre_mapping = create_genre_mapping()
    processed = 0
    skipped = 0
    mapped_counts = {}
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                skipped += 1
                continue
            original_genres = parts[0].lower().split(',')
            content = parts[1]
            mapped = False
            for genre in original_genres:
                genre = cleanup_genre_name(genre.strip())
                if genre in genre_mapping:
                    parent_genre = genre_mapping[genre]
                    outfile.write(f"{parent_genre}\t{content}\n")
                    mapped_counts[parent_genre] = mapped_counts.get(parent_genre, 0) + 1
                    mapped = True
                    processed += 1
            if not mapped:
                skipped += 1
    print(f"\nDataset Mapping Complete!")
    print(f"Total lines processed: {processed:,}")
    print(f"Total lines skipped: {skipped:,}")
    print(f"\nGenre Distribution in Mapped Dataset:")
    print("================================")
    total_samples = sum(mapped_counts.values())
    for genre, count in sorted(mapped_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_samples) * 100
        print(f"{genre:<20}: {count:,} samples ({percentage:.1f}%)")
    return mapped_counts

def analyze_dataset(filepath):
    total_lines = 0
    genre_counts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            genre = parts[0].strip()
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            total_lines += 1
    return {
        'total_lines': total_lines,
        'genre_counts': genre_counts
    }

def add_compound_genres(genre_mapping):
    compound_mappings = {
        'horror comedy': 'Horror',
        'romantic comedy': 'Comedy',
        'action comedy': 'Action',
        'sci-fi action': 'Science Fiction',
        'comedy drama': 'Comedy',
        'drama comedy': 'Comedy',
        'historical drama': 'Historical',
        'crime drama': 'Crime',
        'war drama': 'War',
        'romantic drama': 'Romance',
        'musical comedy': 'Musical',
        'mystery thriller': 'Mystery',
        'crime thriller': 'Crime',
        'science fiction action': 'Science Fiction',
        'action thriller': 'Action',
        'comedy thriller': 'Comedy',
        'war comedy': 'War',
        'western comedy': 'Western',
        'social drama': 'Drama',
        'romantic thriller': 'Romance',
        'political thriller': 'Social',
        'superhero action': 'Action',
        'spy thriller': 'Thriller',
        'martial arts action': 'Action',
        'family drama': 'Family',
        'teen comedy': 'Youth',
        'biographical drama': 'Biography',
        'sports drama': 'Sports',
        'musical drama': 'Musical',
        'costume drama': 'Historical',
        'historical romance': 'Historical',
        'fantasy comedy': 'Fantasy',
        'documentary drama': 'Documentary',
        'social comedy': 'Social',
        'teen drama': 'Youth',
        'romantic fantasy': 'Romance',
        'comedy horror': 'Horror',
        'fantasy action': 'Fantasy',
        'mystery comedy': 'Mystery',
        'historical comedy': 'Historical',
        'gangster drama': 'Crime',
        'social thriller': 'Social',
        'sports comedy': 'Sports',
        'war thriller': 'War',
        'western drama': 'Western',
        'supernatural thriller': 'Horror',
        'comedy western': 'Western',
        'mystery horror': 'Horror',
        'sci-fi horror': 'Science Fiction',
        'crime comedy': 'Crime',
        'political drama': 'Social',
        'western thriller': 'Western',
        'erotic thriller': 'Thriller',
        'biographical comedy': 'Biography',
        'drama thriller': 'Drama',
        'teen romance': 'Youth'
    }
    for genre, parent in compound_mappings.items():
        genre_mapping[genre.lower()] = parent
        genre_mapping[genre.lower().replace(' ', '-')] = parent
        genre_mapping[f"{genre.lower()} film"] = parent
        genre_mapping[genre.lower().replace(' ', '/')] = parent
    return genre_mapping

def cleanup_genre_name(genre):
    suffixes = [
        ' film', ' movie', ' picture', '[not in citation given]', ' production',
        ' drama', ' comedy', ' action', ' thriller', ' horror', ' romance',
        ' musical', ' western', ' fantasy', ' adventure', ' documentary',
        ' animation', ' biography', ' war', ' crime', ' mystery', ' experimental',
        ' short', ' family'
    ]
    if 'biographical' in genre:
        return 'biography'
    if 'martial art' in genre or 'kung fu' in genre:
        return 'action'
    if 'sci-fi' in genre or 'science fiction' in genre:
        return 'science fiction'
    if 'world war' in genre or 'wwii' in genre or 'ww1' in genre:
        return 'war'
    if 'romantic comedy' in genre or 'rom com' in genre or 'romcom' in genre:
        return 'comedy'
    if 'documentary' in genre or 'docudrama' in genre:
        return 'documentary'
    if any(x in genre for x in ['yakuza', 'gangster', 'mob', 'crime']):
        return 'crime'
    if 'historical' in genre or 'period' in genre:
        return 'historical'
    if 'experimental' in genre or 'avant garde' in genre or 'art' in genre:
        return 'experimental'
    if 'animation' in genre or 'animated' in genre or 'anime' in genre:
        return 'animation'
    if 'family' in genre or 'children' in genre:
        return 'family'
    if 'teen' in genre or 'youth' in genre or 'coming of age' in genre:
        return 'youth'
    if 'social' in genre or 'political' in genre:
        return 'social'
    if 'short' in genre:
        return 'short film'
    if 'sport' in genre or 'boxing' in genre or 'wrestling' in genre:
        return 'sports'
    for suffix in suffixes:
        if genre.endswith(suffix):
            genre = genre[:-len(suffix)]
    genre = genre.strip().lower()
    genre = genre.replace('/', ' ').replace('-', ' ').replace('_', ' ')
    genre = ' '.join(genre.split())
    return genre

if __name__ == "__main__":
    input_file = "datasets/english/english_final30k_train_cleaned.txt"
    output_file = "english_final30k_train_mapped.txt"
    print(f"Creating mapped dataset from {input_file} ...")
    mapped_counts = create_mapped_dataset(input_file, output_file)
    print(f"\nAnalyzing mapped dataset {output_file} ...")
    results = analyze_dataset(output_file)
    print("\nGenre Distribution:")
    print("==================")
    genre_counts = results['genre_counts']
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    for genre, count in sorted_genres:
        print(f"{genre:<20}: {count} samples")
    print(f"\nTotal genres: {len(genre_counts)}")
    print(f"Total samples: {results['total_lines']}")
