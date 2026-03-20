require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const vision = require('@google-cloud/vision');

const app = express();
const PORT = process.env.PORT || 3000;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const VISION_API_URL = GOOGLE_API_KEY
  ? `https://vision.googleapis.com/v1/images:annotate?key=${GOOGLE_API_KEY}`
  : null;

const GOOGLE_APPLICATION_CREDENTIALS_JSON = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON;
let visionClient = null;
if (GOOGLE_APPLICATION_CREDENTIALS_JSON) {
  try {
    let credsObj;
    try {
      credsObj = JSON.parse(GOOGLE_APPLICATION_CREDENTIALS_JSON);
    } catch (_) {
      const decoded = Buffer.from(GOOGLE_APPLICATION_CREDENTIALS_JSON, 'base64').toString('utf8');
      credsObj = JSON.parse(decoded);
    }

    visionClient = new vision.ImageAnnotatorClient({
      credentials: credsObj
    });
  } catch (e) {
    console.error('Failed to initialize Google Vision client from GOOGLE_APPLICATION_CREDENTIALS_JSON:', e?.message || e);
    visionClient = null;
  }
}

console.log('Vision auth mode:', visionClient ? 'service_account' : (VISION_API_URL ? 'api_key' : 'not_configured'));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-3.5-turbo';
const OPENAI_MODEL_FALLBACKS = (process.env.OPENAI_MODEL_FALLBACKS || 'gpt-3.5-turbo')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

async function callOpenAIChatCompletionsWithFallback(payload, { preferredModel }) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  const modelsToTry = [preferredModel, ...OPENAI_MODEL_FALLBACKS]
    .filter(Boolean)
    .filter((m, i, arr) => arr.indexOf(m) === i);

  let lastError = null;
  for (const model of modelsToTry) {
    try {
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        { ...payload, model },
        {
          headers: {
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout: 25000
        }
      );
      return response;
    } catch (err) {
      lastError = err;
      const status = err?.response?.status;
      const msg = err?.response?.data?.error?.message || err?.message || '';
      const isModelAccessDenied = status === 403 && /does not have access to model/i.test(msg);
      if (isModelAccessDenied) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error('OpenAI request failed');
}

// Middleware
app.use(cors());
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

// File filter function
const fileFilter = function(req, file, cb) {
  const filetypes = /jpeg|jpg|png|gif/;
  const mimetype = filetypes.test(file.mimetype);
  const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
  
  if (mimetype && extname) {
    return cb(null, true);
  }
  cb(new Error('Only image files are allowed (JPEG, JPG, PNG, GIF)'));
};

const upload = multer({ 
  storage: storage,
  limits: { 
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 1
  },
  fileFilter: fileFilter
});

// Function to analyze image using Google Cloud Vision API
async function analyzeImageWithVision(imageBuffer) {
  try {
    const base64Image = imageBuffer.toString('base64');

    if (visionClient) {
      const [result] = await visionClient.annotateImage({
        image: { content: base64Image },
        features: [{ type: 'TEXT_DETECTION', maxResults: 10 }]
      });
      return { responses: [result] };
    }

    if (!VISION_API_URL) {
      throw new Error('Google Vision credentials are not configured (set GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_API_KEY)');
    }

    const request = {
      requests: [
        {
          image: {
            content: base64Image
          },
          features: [
            {
              type: 'TEXT_DETECTION',
              maxResults: 10
            }
          ]
        }
      ]
    };

    const response = await axios.post(VISION_API_URL, request, {
      timeout: 25000 // 25 second timeout
    });
    return response.data;
  } catch (error) {
    const status = error?.response?.status;
    const data = error?.response?.data;
    console.error('Error analyzing image with Vision API:', {
      message: error?.message,
      status,
      data
    });
    throw error;
  }
}

async function generateOpenAIResponse({ text, spicinessLevel, context }) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  const spicinessStyle = [
    'chill and friendly',
    'smooth and confident',
    'spicy and playful',
    'overload rizz: bold, flirty, and high-energy'
  ][Math.min(Math.max(spicinessLevel, 0), 3)];

  const userContext = (context || '').trim();
  const ocrText = (text || '').trim();
  const promptParts = [
    `You are Reply Rizzler. Generate ONE short reply the user can send.`,
    `Style: ${spicinessStyle}.`,
    `Keep it natural, not cringe, and do not mention AI or OCR.`,
    `If the OCR text looks like a chat conversation, respond to the last message.`,
    userContext ? `Extra context from user: ${userContext}` : null,
    `OCR text from the image:\n${ocrText}`
  ].filter(Boolean);

  const payload = {
    messages: [
      { role: 'system', content: 'You write concise chat replies.' },
      { role: 'user', content: promptParts.join('\n\n') }
    ],
    temperature: 0.8,
    max_tokens: 120
  };

  const response = await callOpenAIChatCompletionsWithFallback(payload, { preferredModel: OPENAI_MODEL });

  const content = response?.data?.choices?.[0]?.message?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('No response from OpenAI');
  }
  return content.trim();
}

// Routes
app.get('/', (req, res) => {
  res.send('Hangly Backend is running!');
});

app.post('/api/glowup', async (req, res) => {
  try {
    const { message, tone, include_emojis } = req.body || {};

    const msg = (message || '').toString().trim();
    const toneStr = (tone || '').toString().trim();
    const includeEmojis = include_emojis === true;

    if (!msg) {
      return res.status(400).json({
        success: false,
        error: 'Message is required',
        message: 'Message is required'
      });
    }

    if (!OPENAI_API_KEY) {
      return res.status(500).json({
        success: false,
        error: 'OPENAI_API_KEY is not configured',
        message: 'OPENAI_API_KEY is not configured'
      });
    }

    const toneInstruction = toneStr ? `Tone: ${toneStr}.` : 'Tone: neutral.';
    const emojiInstruction = includeEmojis
      ? 'You may add a few tasteful emojis where they fit naturally.'
      : 'Do not add any emojis.';

    const prompt = [
      'Improve the user\'s message to be more engaging, clear, and polished.',
      toneInstruction,
      emojiInstruction,
      'Keep the meaning the same. Keep it natural (not cringe).',
      'Return only the improved message, no quotes, no explanations.',
      `User message:\n${msg}`
    ].join('\n\n');

    const payload = {
      messages: [
        { role: 'system', content: 'You rewrite short messages for texting.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 200
    };

    const response = await callOpenAIChatCompletionsWithFallback(payload, { preferredModel: OPENAI_MODEL });

    const content = response?.data?.choices?.[0]?.message?.content;
    if (!content || typeof content !== 'string') {
      throw new Error('No response from OpenAI');
    }

    return res.json({
      success: true,
      improved_message: content.trim(),
      tone: toneStr,
      include_emojis: includeEmojis
    });
  } catch (error) {
    const upstreamStatus = error?.response?.status;
    const upstreamData = error?.response?.data;
    const upstreamMessage =
      upstreamData?.error?.message ||
      upstreamData?.error?.type ||
      upstreamData?.error?.code ||
      (typeof upstreamData === 'string' ? upstreamData : null);

    console.error('Unexpected error in /api/glowup:', {
      message: error?.message,
      upstream_status: upstreamStatus,
      upstream_message: upstreamMessage,
      upstream_data: upstreamData
    });
    return res.status(500).json({
      success: false,
      error: 'Failed to glow up message',
      message: upstreamMessage || error?.message || 'Failed to glow up message',
      details: upstreamData || error?.message,
      upstream_status: upstreamStatus
    });
  }
});

// Analyze image and generate response
app.post('/api/analyze', (req, res) => {
  upload.single('image')(req, res, async (err) => {
    try {
      if (err instanceof multer.MulterError) {
        console.error('Multer error:', err);
        return res.status(400).json({ 
          error: 'File upload error',
          details: err.message 
        });
      } else if (err) {
        console.error('Upload error:', err);
        return res.status(400).json({ 
          error: 'File upload failed',
          details: err.message 
        });
      }

      const { spiciness_level, context } = req.body;
      const imagePath = req.file?.path;

      console.log('Received request with file:', req.file);
      console.log('Spiciness level:', spiciness_level);
      console.log('Context:', context);

      if (!imagePath) {
        return res.status(400).json({ 
          error: 'No image file provided',
          receivedFiles: req.files,
          body: req.body
        });
      }

      try {
        // Read image file
        console.log('Reading image file from:', imagePath);
        const imageBuffer = fs.readFileSync(imagePath);
        
        // Analyze image with Google Cloud Vision
        console.log('Analyzing image with Vision API...');
        const visionResult = await analyzeImageWithVision(imageBuffer);
        
        // Extract labels from vision result
        const labels = visionResult.responses[0]?.labelAnnotations?.map(label => label.description) || [];
        const text = visionResult.responses[0]?.textAnnotations?.[0]?.description || '';
        
        console.log('Detected labels:', labels);
        console.log('Detected text:', text);
        
        // Generate response based on detected content
        const spicinessInt = parseInt(spiciness_level || '0');
        let response;
        if (text && text.trim().length > 0) {
          try {
            response = await generateOpenAIResponse({
              text,
              spicinessLevel: spicinessInt,
              context
            });
          } catch (openAiError) {
            console.error('OpenAI error, falling back to local responses:', openAiError);
            response = generateResponse(labels, text, spicinessInt);
          }
        } else {
          response = generateResponse(labels, text, spicinessInt);
        }
        console.log('Generated response:', response);

        // Clean up the uploaded file
        try {
          if (fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
            console.log('Temporary file deleted:', imagePath);
          }
        } catch (cleanupError) {
          console.error('Error cleaning up file:', cleanupError);
        }

        // Send the response
        res.json({
          success: true,
          response: response,
          spiciness_level: spicinessInt,
          detected_labels: labels,
          detected_text: text
        });

      } catch (processError) {
        console.error('Error processing image:', processError);
        // Clean up the uploaded file in case of error
        if (imagePath && fs.existsSync(imagePath)) {
          fs.unlinkSync(imagePath);
        }

        const upstreamStatus = processError?.response?.status;
        const upstreamData = processError?.response?.data;
        const upstreamMessage =
          upstreamData?.error?.message ||
          upstreamData?.error?.status ||
          (typeof upstreamData === 'string' ? upstreamData : null);
        
        return res.status(500).json({ 
          error: 'Failed to process image',
          message: upstreamMessage || processError.message,
          details: upstreamData || processError.message,
          upstream_status: upstreamStatus,
          stack: processError.stack
        });
      }
    } catch (error) {
      console.error('Unexpected error in /api/analyze:', error);
      res.status(500).json({ 
        error: 'Internal server error',
        message: error.message,
        details: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      });
    }
  });
});

// Generate response based on detected content
function generateResponse(labels, text, spiciness) {
  const spicinessLevels = [
    // Level 0: Mild
    [
      "That's a nice photo!",
      "I like what I see!",
      "Great picture!"
    ],
    // Level 1: Friendly
    [
      "Looking good! I like the composition.",
      "Great shot! The lighting is perfect.",
      "You've got a great eye for photography!"
    ],
    // Level 2: Playful
    [
      "Wow, looking amazing! 😍",
      "This picture is fire! 🔥",
      "Someone's looking extra attractive today! 😉"
    ],
    // Level 3: Flirty
    [
      "🔥 HOT DAMN! You're setting my phone on fire! 🔥",
      "🚨 WARNING: This level of attractiveness should be illegal! 🚨",
      "💘 Is it hot in here or is it just you? 💘"
    ]
  ];

  // Get a random response based on spiciness level
  const responses = spicinessLevels[Math.min(spiciness, 3)] || spicinessLevels[0];
  const randomResponse = responses[Math.floor(Math.random() * responses.length)];
  
  // Add some context if we detected text or labels
  let context = '';
  if (text) {
    context = ` I see some text: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`;
  } else if (labels.length > 0) {
    context = ` I can see ${labels.slice(0, 3).join(', ')}.`;
  }
  
  return `${randomResponse}${context}`;
}

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Something went wrong!',
    message: err.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
