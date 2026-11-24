import LegalLayout from './LegalLayout'

/**
 * Converted from legal/terms.html. The wording is unchanged; only the page
 * around it (LegalLayout: header, document switcher, "On this page") is new.
 */
export default function Terms() {
  return (
    <LegalLayout
      title="Terms of Service"
      effective="Effective Date: December 5, 2025"
      intro={
        <>
          <p>
            Welcome to{' '}
            <strong>
              Grade
              <span style={{ color: 'var(--brand)' }}>Ant</span>
            </strong>
            , an AI-powered homework grading tool provided by GradeAnt, affiliated with Missouri S&T. These Terms of Service govern your access to and use of our platform. By using our service, you agree to these terms.
          </p>
        </>
      }
    >
      <div id="service-description" className="lg-section">
        <h2>
          1. Service Description
        </h2>
        <p>
          GradeAnt provides an AI tool designed to assist university-level students and educational institutions by providing automated feedback and preliminary grades on submitted homework assignments. Our current focus is on physics, with plans to expand to coding and mathematics.
        </p>
        <div className="lg-callout">
          <p>
            <strong>Important:</strong>
            {' '}Submitted homework (PDF format, typed or handwritten) receives instant feedback and a suggested grade. You acknowledge that final grades are determined and assigned solely by the relevant teacher or educational institution.
          </p>
        </div>
      </div>
      <div id="eligibility" className="lg-section">
        <h2>
          2. Eligibility and Account Registration
        </h2>
        <p>
          <strong>Eligibility:</strong>
          {' '}You must be at least 16 years old to use the Service. The Service is intended for university students and authorized personnel from educational institutions. By using the Service, you represent and warrant that you meet these eligibility requirements.
        </p>
        <p>
          <strong>Account:</strong>
          {' '}To access the Service, you must register for an account using a valid email address and specifying your university. You are responsible for maintaining the confidentiality of your account credentials and for all activities that occur under your account. You agree to notify us immediately of any unauthorized use of your account.
        </p>
      </div>
      <div id="use-of-service" className="lg-section">
        <h2>
          3. Use of the Service
        </h2>
        <p>
          <strong>License:</strong>
          {' '}We grant you a limited, non-exclusive, non-transferable, revocable license to access and use the Service for legitimate educational purposes in accordance with these Terms.
        </p>
        <p>
          <strong>Acceptable Use:</strong>
          {' '}You agree to use the Service only for its intended purpose: submitting homework assigned to you (or that you have explicit permission to submit) for feedback and preliminary grading.
        </p>
        <p>
          <strong>Prohibited Use:</strong>
          {' '}You agree not to:
        </p>
        <ul>
          <li>
            Submit work that is not your own or engage in any form of academic dishonesty ("cheating").
          </li>
          <li>
            Attempt to manipulate, deceive, or "trick" the AI grading system (e.g., through prompt injection).
          </li>
          <li>Upload or transmit malicious files, viruses, or code.</li>
          <li>Overload, disrupt, or impair the functioning of the Service.</li>
          <li>
            Reverse-engineer, decompile, or attempt to discover the source code or underlying algorithms of the Service or AI models.
          </li>
          <li>Use the Service for any illegal or unauthorized purpose.</li>
        </ul>
      </div>
      <div id="homework-submission" className="lg-section">
        <h2>
          4. Homework Submission
        </h2>
        <p>Users submit homework assignments directly by uploading PDF files.</p>
        <p>
          The number of submissions may be limited based on the assignments set by the associated educational institution or teacher.
        </p>
      </div>
      <div id="intellectual-property" className="lg-section">
        <h2>
          5. Intellectual Property
        </h2>
        <p>
          <strong>Our IP:</strong>
          {' '}The Service, including the website, design, text, graphics, logos, AI models, grading algorithms, feedback mechanisms, and all underlying software and technology, are the exclusive property of GradeAnt and/or its licensors and is protected by intellectual property laws.
        </p>
        <p>
          <strong>Your IP:</strong>
          {' '}You retain ownership of the original content of the homework you submit ("User Content").
        </p>
        <p>
          <strong>License Grant to Us:</strong>
          {' '}By submitting User Content to the Service, you grant GradeAnt a worldwide, non-exclusive, royalty-free license to:
        </p>
        <ul>
          <li>
            Store, process, analyze, and reproduce your User Content solely for the purpose of providing the grading and feedback Service to you and authorized educational personnel (e.g., your teacher/institution).
          </li>
          <li>
            Display the results (grades, feedback) back to you and authorized educational personnel.
          </li>
          <li>
            Use anonymized or aggregated User Content for research purposes, conducted in accordance with relevant ethical guidelines (such as Institutional Review Board - IRB - standards) and our Privacy Policy.
          </li>
        </ul>
        <div className="lg-callout">
          <p>
            <strong>Note:</strong>
            {' '}Our research use does not currently include training our AI models, but focuses on understanding educational patterns and improving pedagogical approaches. We will take steps to ensure sufficient anonymity is maintained for research use.
          </p>
        </div>
      </div>
      <div id="service-fees" className="lg-section">
        <h2>
          6. Service Fees (Currently Free)
        </h2>
        <p>
          <span className="tooltip">
            The Service is currently provided free of charge{' '}
            <span className="tooltip-text">No payment required at this time</span>
          </span>
          . We reserve the right to introduce fees or subscription models in the future. Any changes to pricing will be communicated to you in advance.
        </p>
      </div>
      <div id="disclaimers" className="lg-section">
        <h2>
          7. Disclaimers
        </h2>
        <div className="lg-callout">
          <p>
            <strong>Accuracy Disclaimer:</strong>
            {' '}THE AI GRADING AND FEEDBACK PROVIDED BY THE SERVICE ARE INTENDED AS EDUCATIONAL TOOLS. WHILE WE STRIVE FOR ACCURACY, THE AI MAY NOT BE PERFECT AND CAN MAKE ERRORS. THE PROVIDED GRADES ARE PRELIMINARY AND SUGGESTIVE ONLY. FINAL ACADEMIC GRADES ARE DETERMINED SOLELY BY YOUR TEACHER OR EDUCATIONAL INSTITUTION.
          </p>
        </div>
        <p>
          <strong>No Guarantee of Success:</strong>
          {' '}GradeAnt makes no representations or warranties regarding the accuracy, reliability, or completeness of the Service. Use of the Service does not guarantee improved grades, academic success, or any specific outcome.
        </p>
        <p>
          <strong>"As Is" Service:</strong>
          {' '}The Service is provided "as is" and "as available" without warranties of any kind, either express or implied, including, but not limited to, implied warranties of merchantability, fitness for a particular purpose, or non-infringement.
        </p>
        <p>
          <strong>Third-Party Links:</strong>
          {' '}The Service may contain links to third-party websites or services that are not owned or controlled by GradeAnt. We are not responsible for the content or practices of any third-party sites.
        </p>
      </div>
      <div id="limitation" className="lg-section">
        <h2>
          8. Limitation of Liability
        </h2>
        <p>
          TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, GRADEANT, MISSOURI S&T, AND THEIR AFFILIATES, OFFICERS, DIRECTORS, EMPLOYEES, AGENTS, AND LICENSORS SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR ANY LOSS OF PROFITS OR REVENUES, WHETHER INCURRED DIRECTLY OR INDIRECTLY, OR ANY LOSS OF DATA, USE, GOODWILL, OR OTHER INTANGIBLE LOSSES.
        </p>
        <p>
          IN NO EVENT SHALL GRADEANT'S AGGREGATE LIABILITY EXCEED ZERO U.S. DOLLARS ($0.00) AS THE SERVICE IS CURRENTLY FREE.
        </p>
      </div>
      <div id="termination" className="lg-section">
        <h2>
          9. Termination
        </h2>
        <p>
          <strong>By Us:</strong>
          {' '}We may suspend or terminate your access to the Service at any time, with or without cause or notice, including but not limited to, if you breach these Terms, engage in prohibited conduct, or if required by law.
        </p>
        <p>
          <strong>By You:</strong>
          {' '}You may stop using the Service and request account deletion at any time by contacting us at contact@gradeant.com
        </p>
        <p>
          <strong>Effect of Termination:</strong>
          {' '}Upon termination, your right to use the Service ceases immediately. We may delete your account data in accordance with our data retention policies (see Privacy Policy). Sections related to Intellectual Property, Disclaimers, Limitation of Liability, Governing Law, and Dispute Resolution shall survive termination.
        </p>
      </div>
      <div id="governing-law" className="lg-section">
        <h2>
          10. Governing Law and Dispute Resolution
        </h2>
        <p>
          These Terms shall be governed by and construed in accordance with the laws of the State of Missouri, without regard to its conflict of law principles. Any dispute arising out of or relating to these Terms or the Service shall be resolved exclusively in the state or federal courts located in Phelps County, Missouri (or the appropriate federal district).
        </p>
      </div>
      <div id="changes" className="lg-section">
        <h2>
          11. Changes to Terms
        </h2>
        <p>
          We reserve the right to modify these Terms at any time. We will notify you of significant changes by sending an email to the address associated with your account or by posting a notice on the Service. Your continued use of the Service after such changes constitutes your acceptance of the new Terms.
        </p>
      </div>
      <div id="contact" className="lg-section">
        <h2>
          12. Contact Information
        </h2>
        <div className="lg-contact">
          <p>
            If you have any questions about these Terms, please contact us at:{' '}
            <strong>contact@gradeant.com</strong>
          </p>
        </div>
      </div>
    </LegalLayout>
  )
}
